"""
Train ARC translation model on Modal GPU.

Usage:
    modal run train_modal.py --quick  # Fast test
    modal run train_modal.py          # Full training
"""

import modal

app = modal.App("arc-primitive-translation")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0",
    "numpy>=1.21",
    "scipy>=1.10",
)

volume = modal.Volume.from_name("arc-translation-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/outputs": volume},
)
def train_experiment(quick: bool = False):
    """Train ARC translation model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import json
    import time
    from pathlib import Path
    from scipy import ndimage
    from collections import Counter
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
    
    # =========================================================================
    # Metrics (proper ARC-style)
    # =========================================================================
    
    def exact_match_rate(preds, targets):
        """% grids that match exactly."""
        matches = np.all(preds == targets, axis=(1, 2))
        return float(np.mean(matches))
    
    def foreground_accuracy(preds, targets):
        """Accuracy over non-background cells only."""
        fg_mask = (preds != 0) | (targets != 0)
        if not np.any(fg_mask):
            return 1.0
        correct = (preds == targets) & fg_mask
        return float(np.sum(correct) / np.sum(fg_mask))
    
    def get_connected_components(grid):
        """Extract connected components from grid."""
        objects = []
        for color in range(1, 10):
            mask = (grid == color).astype(np.int32)
            if not np.any(mask):
                continue
            labeled, n_comp = ndimage.label(mask)
            for comp_id in range(1, n_comp + 1):
                cells = list(zip(*np.where(labeled == comp_id)))
                if cells:
                    objects.append({'color': color, 'cells': cells})
        return objects
    
    def object_iou(pred_grid, target_grid):
        """Object-level IoU."""
        pred_obj = get_connected_components(pred_grid)
        target_obj = get_connected_components(target_grid)
        if not target_obj:
            return 1.0 if not pred_obj else 0.0
        ious = []
        for t in target_obj:
            t_cells = set(t['cells'])
            best_iou = 0.0
            for p in pred_obj:
                if p['color'] != t['color']:
                    continue
                p_cells = set(p['cells'])
                inter = len(t_cells & p_cells)
                union = len(t_cells | p_cells)
                iou = inter / union if union > 0 else 0
                best_iou = max(best_iou, iou)
            ious.append(best_iou)
        return float(np.mean(ious)) if ious else 0.0
    
    def compute_metrics(preds, targets):
        """Compute all metrics."""
        metrics = {
            'exact_match': exact_match_rate(preds, targets),
            'foreground_acc': foreground_accuracy(preds, targets),
            'cell_acc': float(np.mean(preds == targets)),
        }
        # Object IoU (per sample)
        ious = [object_iou(preds[i], targets[i]) for i in range(len(preds))]
        metrics['object_iou'] = float(np.mean(ious))
        return metrics
    
    # =========================================================================
    # Compiled Solver (Classical Baseline)
    # =========================================================================
    
    def infer_translation(input_grid, output_grid):
        """Infer (dx, dy) by matching objects."""
        in_obj = get_connected_components(input_grid)
        out_obj = get_connected_components(output_grid)
        if not in_obj or not out_obj:
            return None
        votes = {}
        for i_o in in_obj:
            i_cells = set(i_o['cells'])
            for o_o in out_obj:
                if o_o['color'] != i_o['color']:
                    continue
                o_cells = set(o_o['cells'])
                if abs(len(i_cells) - len(o_cells)) > 2:
                    continue
                i_cent = np.mean(list(i_cells), axis=0)
                o_cent = np.mean(list(o_cells), axis=0)
                dx, dy = int(round(o_cent[0] - i_cent[0])), int(round(o_cent[1] - i_cent[1]))
                votes[(dx, dy)] = votes.get((dx, dy), 0) + 1
        if not votes:
            return None
        return max(votes.keys(), key=lambda k: votes[k])
    
    def apply_translation(grid, dx, dy):
        """Apply translation to all objects."""
        H, W = grid.shape
        objects = get_connected_components(grid)
        new_grid = np.zeros_like(grid)
        for obj in objects:
            for r, c in obj['cells']:
                nr, nc = r + dx, c + dy
                if 0 <= nr < H and 0 <= nc < W:
                    new_grid[nr, nc] = obj['color']
        return new_grid
    
    def compiled_solver_batch(inputs, targets):
        """Run compiled solver on a batch."""
        preds = np.zeros_like(inputs)
        for i in range(len(inputs)):
            t = infer_translation(inputs[i], targets[i])
            if t:
                preds[i] = apply_translation(inputs[i], t[0], t[1])
            else:
                preds[i] = inputs[i]
        return preds
    
    # =========================================================================
    # Data Generation - ARC-style Conditioning
    # =========================================================================
    # Each task has: shared (dx,dy), multiple train pairs, one test pair
    # Model sees (train_inputs, train_outputs, test_input) -> predicts test_output
    # This makes dx,dy inferable from context!
    
    def generate_polyomino(size, rng):
        if size <= 0:
            return []
        cells = [(0, 0)]
        while len(cells) < size:
            base = cells[rng.integers(len(cells))]
            neighbors = [(base[0]-1, base[1]), (base[0]+1, base[1]),
                        (base[0], base[1]-1), (base[0], base[1]+1)]
            rng.shuffle(neighbors)
            for n in neighbors:
                if n not in cells:
                    cells.append(n)
                    break
        return cells
    
    def normalize_polyomino(cells):
        if not cells:
            return cells
        min_r = min(c[0] for c in cells)
        min_c = min(c[1] for c in cells)
        return [(r - min_r, c - min_c) for r, c in cells]
    
    def get_bounding_box(cells):
        if not cells:
            return (0, 0, 0, 0)
        rows = [c[0] for c in cells]
        cols = [c[1] for c in cells]
        return (min(rows), min(cols), max(rows), max(cols))
    
    def can_place_object(grid, cells, top_left, min_gap=1):
        H, W = grid.shape
        r0, c0 = top_left
        for dr, dc in cells:
            r, c = r0 + dr, c0 + dc
            if r < 0 or r >= H or c < 0 or c >= W:
                return False
            if grid[r, c] != 0:
                return False
            for gr in range(max(0, r - min_gap), min(H, r + min_gap + 1)):
                for gc in range(max(0, c - min_gap), min(W, c + min_gap + 1)):
                    if (gr, gc) != (r, c) and grid[gr, gc] != 0:
                        if (gr - r0, gc - c0) not in cells:
                            return False
        return True
    
    def place_object(grid, cells, top_left, color):
        r0, c0 = top_left
        for dr, dc in cells:
            grid[r0 + dr, c0 + dc] = color
    
    def generate_scene(grid_size, n_objects, rng, flag_prob=0.3, line_prob=0.2):
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        objects = []
        available_colors = list(range(1, 10))
        rng.shuffle(available_colors)
        
        for _ in range(n_objects):
            if not available_colors:
                break
            size = rng.integers(3, 9)
            cells = normalize_polyomino(generate_polyomino(size, rng))
            color = available_colors.pop()
            bbox = get_bounding_box(cells)
            obj_h = bbox[2] - bbox[0] + 1
            obj_w = bbox[3] - bbox[1] + 1
            
            for _ in range(100):
                r = rng.integers(0, max(1, grid_size - obj_h))
                c = rng.integers(0, max(1, grid_size - obj_w))
                if can_place_object(grid, cells, (r, c)):
                    place_object(grid, cells, (r, c), color)
                    objects.append({'cells': cells, 'color': color, 'top_left': (r, c)})
                    break
        
        # Single-cell flags
        if rng.random() < flag_prob and available_colors:
            n_flags = rng.integers(1, min(4, len(available_colors) + 1))
            for _ in range(n_flags):
                if not available_colors:
                    break
                color = available_colors.pop()
                for _ in range(50):
                    r, c = rng.integers(0, grid_size), rng.integers(0, grid_size)
                    if can_place_object(grid, [(0, 0)], (r, c)):
                        grid[r, c] = color
                        objects.append({'cells': [(0, 0)], 'color': color, 'top_left': (r, c)})
                        break
        
        # Thin lines
        if rng.random() < line_prob and available_colors:
            color = available_colors.pop()
            horizontal = rng.random() < 0.5
            length = rng.integers(3, 7)
            cells = [(0, i) for i in range(length)] if horizontal else [(i, 0) for i in range(length)]
            bbox = get_bounding_box(cells)
            obj_h, obj_w = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            
            for _ in range(50):
                r = rng.integers(0, max(1, grid_size - obj_h))
                c = rng.integers(0, max(1, grid_size - obj_w))
                if can_place_object(grid, cells, (r, c)):
                    place_object(grid, cells, (r, c), color)
                    objects.append({'cells': cells, 'color': color, 'top_left': (r, c)})
                    break
        
        return grid, objects
    
    def get_valid_translations(grid, objects, max_t):
        """Get all valid translations for a scene."""
        H, W = grid.shape
        all_cells = []
        for obj in objects:
            r0, c0 = obj['top_left']
            for dr, dc in obj['cells']:
                all_cells.append((r0 + dr, c0 + dc))
        
        if not all_cells:
            return []
        
        min_r, max_r = min(c[0] for c in all_cells), max(c[0] for c in all_cells)
        min_c, max_c = min(c[1] for c in all_cells), max(c[1] for c in all_cells)
        
        dx_min, dx_max = max(-max_t, -min_r), min(max_t, H - 1 - max_r)
        dy_min, dy_max = max(-max_t, -min_c), min(max_t, W - 1 - max_c)
        
        return [(dx, dy) for dx in range(dx_min, dx_max + 1) 
                for dy in range(dy_min, dy_max + 1) if dx != 0 or dy != 0]
    
    def translate_scene(grid, objects, dx, dy):
        H, W = grid.shape
        new_grid = np.zeros_like(grid)
        for obj in objects:
            r0, c0 = obj['top_left']
            for dr, dc in obj['cells']:
                new_r, new_c = r0 + dr + dx, c0 + dc + dy
                if 0 <= new_r < H and 0 <= new_c < W:
                    new_grid[new_r, new_c] = obj['color']
        return new_grid
    
    def generate_arc_task(grid_size, n_objects_range, n_train_pairs, rng):
        """
        Generate one ARC-style task:
        - Sample a single (dx, dy) for the whole task
        - Generate n_train_pairs train examples + 1 test example
        - All share the same translation rule
        
        Returns:
            train_inputs: (n_train, H, W)
            train_outputs: (n_train, H, W)
            test_input: (H, W)
            test_output: (H, W)
            translation: (dx, dy)
        """
        max_attempts = 50
        
        for _ in range(max_attempts):
            # Sample translation for this task
            dx = rng.integers(-3, 4)
            dy = rng.integers(-3, 4)
            if dx == 0 and dy == 0:
                continue
            
            train_inputs = []
            train_outputs = []
            
            # Generate train pairs
            success = True
            for _ in range(n_train_pairs):
                n_objects = rng.integers(n_objects_range[0], n_objects_range[1] + 1)
                
                # Try to generate a scene where this translation is valid
                for _ in range(20):
                    scene, objects = generate_scene(grid_size, n_objects, rng)
                    if not objects:
                        continue
                    
                    valid_trans = get_valid_translations(scene, objects, 3)
                    if (dx, dy) in valid_trans:
                        output = translate_scene(scene, objects, dx, dy)
                        train_inputs.append(scene)
                        train_outputs.append(output)
                        break
                else:
                    success = False
                    break
            
            if not success or len(train_inputs) < n_train_pairs:
                continue
            
            # Generate test example with same translation
            n_objects = rng.integers(n_objects_range[0], n_objects_range[1] + 1)
            for _ in range(20):
                scene, objects = generate_scene(grid_size, n_objects, rng)
                if not objects:
                    continue
                valid_trans = get_valid_translations(scene, objects, 3)
                if (dx, dy) in valid_trans:
                    test_input = scene
                    test_output = translate_scene(scene, objects, dx, dy)
                    
                    return (
                        np.stack(train_inputs),
                        np.stack(train_outputs),
                        test_input,
                        test_output,
                        (dx, dy)
                    )
        
        return None  # Failed to generate task
    
    def generate_batch_arc_style(batch_size, grid_size, n_objects_range, n_train_pairs, rng):
        """
        Generate a batch of ARC-style tasks.
        
        Returns:
            context: (batch, n_train_pairs * 2 + 1, H, W) - stacked [train_in, train_out, ..., test_in]
            targets: (batch, H, W) - test outputs
            translations: list of (dx, dy) per task
        """
        # Context channels: train_in1, train_out1, train_in2, train_out2, test_in
        n_context = n_train_pairs * 2 + 1
        
        context = np.zeros((batch_size, n_context, grid_size, grid_size), dtype=np.int64)
        targets = np.zeros((batch_size, grid_size, grid_size), dtype=np.int64)
        translations = []
        
        for i in range(batch_size):
            task = generate_arc_task(grid_size, n_objects_range, n_train_pairs, rng)
            if task is None:
                # Fallback: empty task
                translations.append((0, 0))
                continue
            
            train_in, train_out, test_in, test_out, trans = task
            
            # Stack context: [train_in_1, train_out_1, train_in_2, train_out_2, test_in]
            for j in range(n_train_pairs):
                context[i, j * 2] = train_in[j]
                context[i, j * 2 + 1] = train_out[j]
            context[i, -1] = test_in
            
            targets[i] = test_out
            translations.append(trans)
        
        return context, targets, translations
    
    # =========================================================================
    # Model (PyTorch) - Context-Conditioned for ARC-style tasks
    # =========================================================================
    # Input: (batch, n_context, H, W) where n_context = 2*n_train_pairs + 1
    # Context grids are: [train_in_1, train_out_1, ..., test_in]
    # Output: (batch, H, W, n_colors) logits for test_output
    
    class RelativePositionBias2D(nn.Module):
        def __init__(self, n_heads, max_offset):
            super().__init__()
            self.n_heads = n_heads
            self.max_offset = max_offset
            n_buckets = 2 * max_offset + 1
            self.row_bias = nn.Parameter(torch.zeros(n_heads, n_buckets))
            self.col_bias = nn.Parameter(torch.zeros(n_heads, n_buckets))
        
        def forward(self, H, W, device):
            seq_len = H * W
            positions = torch.arange(seq_len, device=device)
            rows, cols = positions // W, positions % W
            
            rel_rows = rows[:, None] - rows[None, :]
            rel_cols = cols[:, None] - cols[None, :]
            
            rel_rows = torch.clamp(rel_rows, -self.max_offset, self.max_offset) + self.max_offset
            rel_cols = torch.clamp(rel_cols, -self.max_offset, self.max_offset) + self.max_offset
            
            return self.row_bias[:, rel_rows] + self.col_bias[:, rel_cols]
    
    class ARCTransformerConditioned(nn.Module):
        """
        Transformer that takes context (train pairs) + test input.
        
        Architecture: Channel-stacked context approach
        - Each cell gets features from all context grids
        - Embeddings for: color (10) × grid_type (5: train_in, train_out, test_in)
        - Cross-attention between positions to infer the rule
        """
        def __init__(self, n_colors=10, n_context=5, d_model=128, n_heads=4, 
                     n_layers=4, d_ff=256, max_offset=8):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_context = n_context
            
            # Color embedding
            self.color_embed = nn.Embedding(n_colors, d_model // 2)
            
            # Grid type embedding (which context grid is this from)
            self.grid_type_embed = nn.Embedding(n_context, d_model // 2)
            
            # Project stacked embeddings to d_model
            self.input_proj = nn.Linear(n_context * d_model, d_model)
            
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'ln1': nn.LayerNorm(d_model),
                    'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                    'rel_pos': RelativePositionBias2D(n_heads, max_offset),
                    'ln2': nn.LayerNorm(d_model),
                    'mlp': nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model),
                    ),
                })
                for _ in range(n_layers)
            ])
            
            self.ln_final = nn.LayerNorm(d_model)
            self.output_proj = nn.Linear(d_model, n_colors)
        
        def forward(self, context):
            """
            context: (batch, n_context, H, W) - stacked grids with color indices
            Returns: (batch, H, W, n_colors) - logits for test output
            """
            batch, n_ctx, H, W = context.shape
            seq_len = H * W
            
            # Embed each context grid separately
            # context: (batch, n_ctx, H, W)
            # Flatten spatial: (batch, n_ctx, H*W)
            context_flat = context.view(batch, n_ctx, seq_len)
            
            # Color embeddings: (batch, n_ctx, seq_len, d_model//2)
            color_emb = self.color_embed(context_flat)
            
            # Grid type embeddings: (batch, n_ctx, seq_len, d_model//2)
            grid_types = torch.arange(n_ctx, device=context.device)
            grid_type_emb = self.grid_type_embed(grid_types)  # (n_ctx, d_model//2)
            grid_type_emb = grid_type_emb[None, :, None, :].expand(batch, -1, seq_len, -1)
            
            # Combine color + grid type
            combined = torch.cat([color_emb, grid_type_emb], dim=-1)  # (batch, n_ctx, seq_len, d_model)
            
            # Stack all context grids for each position
            # (batch, n_ctx, seq_len, d_model) -> (batch, seq_len, n_ctx * d_model)
            combined = combined.permute(0, 2, 1, 3).reshape(batch, seq_len, n_ctx * self.d_model)
            
            # Project to model dimension
            h = self.input_proj(combined)  # (batch, seq_len, d_model)
            
            # Transformer blocks
            for block in self.blocks:
                h_norm = block['ln1'](h)
                
                # Attention with relative position bias
                rel_bias = block['rel_pos'](H, W, context.device)
                
                # Manual attention with bias
                Q = h_norm @ block['attn'].in_proj_weight[:self.d_model].T + block['attn'].in_proj_bias[:self.d_model]
                K = h_norm @ block['attn'].in_proj_weight[self.d_model:2*self.d_model].T + block['attn'].in_proj_bias[self.d_model:2*self.d_model]
                V = h_norm @ block['attn'].in_proj_weight[2*self.d_model:].T + block['attn'].in_proj_bias[2*self.d_model:]
                
                d_head = self.d_model // self.n_heads
                Q = Q.view(batch, seq_len, self.n_heads, d_head).transpose(1, 2)
                K = K.view(batch, seq_len, self.n_heads, d_head).transpose(1, 2)
                V = V.view(batch, seq_len, self.n_heads, d_head).transpose(1, 2)
                
                scores = (Q @ K.transpose(-2, -1)) / (d_head ** 0.5)
                scores = scores + rel_bias[None, :, :, :]
                
                attn = F.softmax(scores, dim=-1)
                out = (attn @ V).transpose(1, 2).reshape(batch, seq_len, self.d_model)
                out = out @ block['attn'].out_proj.weight.T + block['attn'].out_proj.bias
                
                h = h + out
                h = h + block['mlp'](block['ln2'](h))
            
            h = self.ln_final(h)
            logits = self.output_proj(h)
            
            return logits.view(batch, H, W, -1)
    
    
    class ARCTransformerWithTaps(nn.Module):
        """
        Transformer with intermediate "tap heads" for auxiliary supervision.
        
        Key idea: Force the network to maintain separable intermediate representations
        by adding prediction heads at intermediate layers:
        - dxdy_head: predicts the translation vector (forces rule extraction)
        - mask_head: predicts foreground mask (forces object segmentation)
        
        This tests the hypothesis that "more modular supervision" beats 
        "scale the monolith" for OOD generalization.
        """
        def __init__(self, n_colors=10, n_context=5, d_model=128, n_heads=4, 
                     n_layers=4, d_ff=256, max_offset=8, tap_layer=2):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_context = n_context
            self.n_layers = n_layers
            self.tap_layer = tap_layer  # Which layer to tap (0-indexed)
            
            # Color embedding
            self.color_embed = nn.Embedding(n_colors, d_model // 2)
            
            # Grid type embedding
            self.grid_type_embed = nn.Embedding(n_context, d_model // 2)
            
            # Project stacked embeddings to d_model
            self.input_proj = nn.Linear(n_context * d_model, d_model)
            
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'ln1': nn.LayerNorm(d_model),
                    'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                    'rel_pos': RelativePositionBias2D(n_heads, max_offset),
                    'ln2': nn.LayerNorm(d_model),
                    'mlp': nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model),
                    ),
                })
                for _ in range(n_layers)
            ])
            
            self.ln_final = nn.LayerNorm(d_model)
            self.output_proj = nn.Linear(d_model, n_colors)
            
            # =====================
            # TAP HEADS (auxiliary)
            # =====================
            
            # dxdy head: predict translation vector from pooled intermediate state
            # Output: 7 classes for dx (-3 to +3) + 7 classes for dy
            self.dxdy_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 14),  # 7 + 7 for dx, dy
            )
            
            # mask head: predict foreground at each position from intermediate state
            # Output: 1 logit per position (binary: foreground or not)
            self.mask_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
        
        def forward(self, context, return_taps=False):
            """
            context: (batch, n_context, H, W)
            return_taps: if True, also return auxiliary predictions
            
            Returns: 
                logits: (batch, H, W, n_colors)
                taps (optional): dict with 'dxdy_logits', 'mask_logits'
            """
            batch, n_ctx, H, W = context.shape
            seq_len = H * W
            
            # Embed grid tokens
            context_flat = context.view(batch, n_ctx, seq_len)
            color_emb = self.color_embed(context_flat)
            
            grid_types = torch.arange(n_ctx, device=context.device)
            grid_type_emb = self.grid_type_embed(grid_types)
            grid_type_emb = grid_type_emb[None, :, None, :].expand(batch, -1, seq_len, -1)
            
            combined = torch.cat([color_emb, grid_type_emb], dim=-1)
            combined = combined.permute(0, 2, 1, 3).reshape(batch, seq_len, n_ctx * self.d_model)
            h = self.input_proj(combined)
            
            taps = {}
            
            # Transformer blocks with tap
            for layer_idx, block in enumerate(self.blocks):
                h_norm = block['ln1'](h)
                rel_bias = block['rel_pos'](H, W, context.device)
                
                Q = h_norm @ block['attn'].in_proj_weight[:self.d_model].T + block['attn'].in_proj_bias[:self.d_model]
                K = h_norm @ block['attn'].in_proj_weight[self.d_model:2*self.d_model].T + block['attn'].in_proj_bias[self.d_model:2*self.d_model]
                V = h_norm @ block['attn'].in_proj_weight[2*self.d_model:].T + block['attn'].in_proj_bias[2*self.d_model:]
                
                d_head = self.d_model // self.n_heads
                Q = Q.view(batch, seq_len, self.n_heads, d_head).transpose(1, 2)
                K = K.view(batch, seq_len, self.n_heads, d_head).transpose(1, 2)
                V = V.view(batch, seq_len, self.n_heads, d_head).transpose(1, 2)
                
                scores = (Q @ K.transpose(-2, -1)) / (d_head ** 0.5)
                scores = scores + rel_bias[None, :, :, :]
                
                attn = F.softmax(scores, dim=-1)
                out = (attn @ V).transpose(1, 2).reshape(batch, seq_len, self.d_model)
                out = out @ block['attn'].out_proj.weight.T + block['attn'].out_proj.bias
                
                h = h + out
                h = h + block['mlp'](block['ln2'](h))
                
                # TAP at specified layer
                if layer_idx == self.tap_layer:
                    # dxdy: pool over sequence, predict translation
                    h_pooled = h.mean(dim=1)  # (batch, d_model)
                    dxdy_logits = self.dxdy_head(h_pooled)  # (batch, 14)
                    taps['dxdy_logits'] = dxdy_logits
                    
                    # mask: per-position foreground prediction
                    mask_logits = self.mask_head(h).squeeze(-1)  # (batch, seq_len)
                    taps['mask_logits'] = mask_logits.view(batch, H, W)
            
            h = self.ln_final(h)
            logits = self.output_proj(h)
            
            if return_taps:
                return logits.view(batch, H, W, -1), taps
            return logits.view(batch, H, W, -1)


    class ARCTransformerWithMemory(nn.Module):
        """
        Transformer with M learnable memory/slot tokens ("lanes").
        
        These memory tokens:
        - Are prepended to the sequence
        - Can attend to all grid positions and other memory tokens
        - Provide writable workspace for intermediate computation
        - Are discarded at output (only grid positions produce logits)
        
        Hypothesis: Memory tokens should help with object-count OOD scaling
        by providing explicit "slots" for tracking objects.
        """
        def __init__(self, n_colors=10, n_context=5, d_model=128, n_heads=4, 
                     n_layers=4, d_ff=256, max_offset=8, n_memory=8):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_context = n_context
            self.n_memory = n_memory
            
            # Color embedding
            self.color_embed = nn.Embedding(n_colors, d_model // 2)
            
            # Grid type embedding
            self.grid_type_embed = nn.Embedding(n_context, d_model // 2)
            
            # Project stacked embeddings to d_model
            self.input_proj = nn.Linear(n_context * d_model, d_model)
            
            # Learnable memory tokens (slots)
            self.memory_tokens = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)
            
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'ln1': nn.LayerNorm(d_model),
                    'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                    'rel_pos': RelativePositionBias2D(n_heads, max_offset),
                    'ln2': nn.LayerNorm(d_model),
                    'mlp': nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model),
                    ),
                })
                for _ in range(n_layers)
            ])
            
            self.ln_final = nn.LayerNorm(d_model)
            self.output_proj = nn.Linear(d_model, n_colors)
        
        def forward(self, context):
            """
            context: (batch, n_context, H, W)
            Returns: (batch, H, W, n_colors)
            """
            batch, n_ctx, H, W = context.shape
            seq_len = H * W
            
            # Embed grid tokens
            context_flat = context.view(batch, n_ctx, seq_len)
            color_emb = self.color_embed(context_flat)
            
            grid_types = torch.arange(n_ctx, device=context.device)
            grid_type_emb = self.grid_type_embed(grid_types)
            grid_type_emb = grid_type_emb[None, :, None, :].expand(batch, -1, seq_len, -1)
            
            combined = torch.cat([color_emb, grid_type_emb], dim=-1)
            combined = combined.permute(0, 2, 1, 3).reshape(batch, seq_len, n_ctx * self.d_model)
            grid_tokens = self.input_proj(combined)  # (batch, seq_len, d_model)
            
            # Prepend memory tokens
            memory = self.memory_tokens.unsqueeze(0).expand(batch, -1, -1)  # (batch, n_memory, d_model)
            h = torch.cat([memory, grid_tokens], dim=1)  # (batch, n_memory + seq_len, d_model)
            
            total_len = self.n_memory + seq_len
            
            # Transformer blocks
            for block in self.blocks:
                h_norm = block['ln1'](h)
                
                # Build attention bias
                # Memory tokens have no spatial position, so they get 0 bias
                # Grid tokens use 2D relative bias among themselves
                rel_bias_grid = block['rel_pos'](H, W, context.device)  # (n_heads, seq_len, seq_len)
                
                # Full attention bias: (n_heads, total_len, total_len)
                # Memory-to-memory: 0, Memory-to-grid: 0, Grid-to-memory: 0, Grid-to-grid: rel_bias
                full_bias = torch.zeros(self.n_heads, total_len, total_len, device=context.device)
                full_bias[:, self.n_memory:, self.n_memory:] = rel_bias_grid
                
                # Manual attention
                Q = h_norm @ block['attn'].in_proj_weight[:self.d_model].T + block['attn'].in_proj_bias[:self.d_model]
                K = h_norm @ block['attn'].in_proj_weight[self.d_model:2*self.d_model].T + block['attn'].in_proj_bias[self.d_model:2*self.d_model]
                V = h_norm @ block['attn'].in_proj_weight[2*self.d_model:].T + block['attn'].in_proj_bias[2*self.d_model:]
                
                d_head = self.d_model // self.n_heads
                Q = Q.view(batch, total_len, self.n_heads, d_head).transpose(1, 2)
                K = K.view(batch, total_len, self.n_heads, d_head).transpose(1, 2)
                V = V.view(batch, total_len, self.n_heads, d_head).transpose(1, 2)
                
                scores = (Q @ K.transpose(-2, -1)) / (d_head ** 0.5)
                scores = scores + full_bias[None, :, :, :]
                
                attn = F.softmax(scores, dim=-1)
                out = (attn @ V).transpose(1, 2).reshape(batch, total_len, self.d_model)
                out = out @ block['attn'].out_proj.weight.T + block['attn'].out_proj.bias
                
                h = h + out
                h = h + block['mlp'](block['ln2'](h))
            
            # Extract grid tokens (discard memory tokens)
            h_grid = h[:, self.n_memory:, :]  # (batch, seq_len, d_model)
            
            h_grid = self.ln_final(h_grid)
            logits = self.output_proj(h_grid)
            
            return logits.view(batch, H, W, -1)

    # =========================================================================
    # Training Helper
    # =========================================================================
    
    def train_and_eval_model(model, model_name, n_epochs, train_steps_per_epoch, 
                             val_steps, batch_size, grid_size, n_objects_range, 
                             n_train_pairs, lr, rng, use_taps=False, 
                             lambda_dxdy=0.5, lambda_mask=0.3):
        """Train a model and return results.
        
        If use_taps=True, expects model to support return_taps=True and adds
        auxiliary losses for dxdy prediction and foreground mask.
        """
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        if use_taps:
            print(f"  Using tap heads: lambda_dxdy={lambda_dxdy}, lambda_mask={lambda_mask}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, n_epochs * train_steps_per_epoch
        )
        
        history = []
        best_fg_acc = 0  # Fixed: was misleadingly named best_exact_match
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            train_grid_loss = 0
            train_dxdy_loss = 0
            train_mask_loss = 0
            train_dxdy_correct = 0
            train_correct = 0
            train_total = 0
            train_fg_pred = 0
            train_fg_target = 0
            
            for step in range(train_steps_per_epoch):
                context, targets, translations = generate_batch_arc_style(
                    batch_size, grid_size, n_objects_range, n_train_pairs, rng
                )
                context = torch.from_numpy(context).to(device)
                targets = torch.from_numpy(targets).to(device)
                
                optimizer.zero_grad()
                
                if use_taps:
                    logits, taps = model(context, return_taps=True)
                    
                    # Main grid loss
                    grid_loss = F.cross_entropy(logits.view(-1, 10), targets.view(-1))
                    
                    # dxdy auxiliary loss
                    # translations is list of (dx, dy), dx/dy in [-3, 3]
                    # Convert to class indices: dx+3 for first 7, dy+3 for next 7
                    dx_targets = torch.tensor([t[0] + 3 for t in translations], device=device)
                    dy_targets = torch.tensor([t[1] + 3 for t in translations], device=device)
                    
                    dxdy_logits = taps['dxdy_logits']  # (batch, 14)
                    dx_loss = F.cross_entropy(dxdy_logits[:, :7], dx_targets)
                    dy_loss = F.cross_entropy(dxdy_logits[:, 7:], dy_targets)
                    dxdy_loss = (dx_loss + dy_loss) / 2
                    
                    # Track dxdy accuracy
                    dx_pred = dxdy_logits[:, :7].argmax(-1)
                    dy_pred = dxdy_logits[:, 7:].argmax(-1)
                    train_dxdy_correct += ((dx_pred == dx_targets) & (dy_pred == dy_targets)).sum().item()
                    
                    # mask auxiliary loss (foreground prediction)
                    # Target: binary mask where target != 0
                    mask_targets = (targets != 0).float()  # (batch, H, W)
                    mask_logits = taps['mask_logits']  # (batch, H, W)
                    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_targets)
                    
                    # Combined loss
                    loss = grid_loss + lambda_dxdy * dxdy_loss + lambda_mask * mask_loss
                    
                    train_grid_loss += grid_loss.item()
                    train_dxdy_loss += dxdy_loss.item()
                    train_mask_loss += mask_loss.item()
                else:
                    logits = model(context)
                    loss = F.cross_entropy(logits.view(-1, 10), targets.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                preds = logits.argmax(-1)
                train_loss += loss.item()
                train_correct += (preds == targets).sum().item()
                train_total += targets.numel()
                train_fg_pred += (preds != 0).sum().item()
                train_fg_target += (targets != 0).sum().item()
            
            # Validation
            model.eval()
            val_preds_all = []
            val_targets_all = []
            val_dxdy_correct = 0
            val_dxdy_total = 0
            
            with torch.no_grad():
                for _ in range(val_steps):
                    context, targets, translations = generate_batch_arc_style(
                        batch_size, grid_size, n_objects_range, n_train_pairs, rng
                    )
                    context_t = torch.from_numpy(context).to(device)
                    
                    if use_taps:
                        logits, taps = model(context_t, return_taps=True)
                        # Check dxdy accuracy
                        dx_targets = torch.tensor([t[0] + 3 for t in translations], device=device)
                        dy_targets = torch.tensor([t[1] + 3 for t in translations], device=device)
                        dx_pred = taps['dxdy_logits'][:, :7].argmax(-1)
                        dy_pred = taps['dxdy_logits'][:, 7:].argmax(-1)
                        val_dxdy_correct += ((dx_pred == dx_targets) & (dy_pred == dy_targets)).sum().item()
                        val_dxdy_total += len(translations)
                    else:
                        logits = model(context_t)
                    
                    preds = logits.argmax(-1).cpu().numpy()
                    val_preds_all.append(preds)
                    val_targets_all.append(targets)
            
            val_preds = np.concatenate(val_preds_all)
            val_targets = np.concatenate(val_targets_all)
            val_metrics = compute_metrics(val_preds, val_targets)
            
            train_loss /= train_steps_per_epoch
            train_acc = train_correct / train_total
            fg_ratio = train_fg_pred / max(train_fg_target, 1)
            
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_exact_match': val_metrics['exact_match'],
                'val_fg_acc': val_metrics['foreground_acc'],
                'val_obj_iou': val_metrics['object_iou'],
                'fg_ratio': fg_ratio,
            }
            
            if use_taps:
                dxdy_acc = train_dxdy_correct / (train_steps_per_epoch * batch_size)
                val_dxdy_acc = val_dxdy_correct / val_dxdy_total if val_dxdy_total > 0 else 0
                epoch_data['train_dxdy_acc'] = dxdy_acc
                epoch_data['val_dxdy_acc'] = val_dxdy_acc
                epoch_data['train_grid_loss'] = train_grid_loss / train_steps_per_epoch
                epoch_data['train_dxdy_loss'] = train_dxdy_loss / train_steps_per_epoch
                epoch_data['train_mask_loss'] = train_mask_loss / train_steps_per_epoch
                
                print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | "
                      f"Exact: {val_metrics['exact_match']:.3f} | "
                      f"FG: {val_metrics['foreground_acc']:.3f} | "
                      f"dxdy: {val_dxdy_acc:.3f}")
            else:
                print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | "
                      f"Exact: {val_metrics['exact_match']:.3f} | "
                      f"FG: {val_metrics['foreground_acc']:.3f} | "
                      f"IoU: {val_metrics['object_iou']:.3f}")
            
            history.append(epoch_data)
            
            # Save checkpoint - use fg_acc as criterion (more gradual than exact_match)
            # Also always save last epoch as fallback
            if val_metrics['foreground_acc'] > best_fg_acc:
                best_fg_acc = val_metrics['foreground_acc']
                torch.save(model.state_dict(), f"/outputs/best_{model_name}.pt")
            
            # Always save last epoch as fallback
            torch.save(model.state_dict(), f"/outputs/last_{model_name}.pt")
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Best fg_acc: {best_fg_acc:.4f}")
        
        return {
            'best_fg_acc': best_fg_acc,
            'history': history,
            'elapsed_time': elapsed,
            'n_params': n_params,
        }
    
    def eval_model_ood(model, model_name, val_steps, batch_size, grid_size, 
                       n_objects_range, n_train_pairs, rng):
        """Evaluate model on OOD conditions."""
        
        # Try best checkpoint, fall back to last
        checkpoint_path = f"/outputs/best_{model_name}.pt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = f"/outputs/last_{model_name}.pt"
        
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        def get_preds(context_np):
            context_t = torch.from_numpy(context_np).to(device)
            with torch.no_grad():
                logits = model(context_t)
            return logits.argmax(-1).cpu().numpy()
        
        ood_results = {}
        
        # Grid size OOD
        print(f"\n--- {model_name}: Grid Size OOD ---")
        for test_size in [16, 24, 32]:
            all_preds, all_targets, all_inputs = [], [], []
            for _ in range(val_steps):
                context, targets, _ = generate_batch_arc_style(
                    batch_size, test_size, n_objects_range, n_train_pairs, rng
                )
                preds = get_preds(context)
                all_preds.append(preds)
                all_targets.append(targets)
                all_inputs.append(context[:, -1])
            
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            inputs = np.concatenate(all_inputs)
            
            metrics = compute_metrics(preds, targets)
            solver_preds = compiled_solver_batch(inputs, targets)
            solver_metrics = compute_metrics(solver_preds, targets)
            
            ood_results[f"grid_{test_size}x{test_size}"] = {
                'model': metrics, 'compiled_solver': solver_metrics
            }
            print(f"  {test_size}x{test_size}: exact={metrics['exact_match']:.3f}, "
                  f"fg={metrics['foreground_acc']:.3f}")
        
        # Object count OOD
        print(f"\n--- {model_name}: Object Count OOD ---")
        for n_obj_range in [(1, 4), (5, 7), (8, 10)]:
            all_preds, all_targets, all_inputs = [], [], []
            for _ in range(val_steps):
                context, targets, _ = generate_batch_arc_style(
                    batch_size, grid_size, n_obj_range, n_train_pairs, rng
                )
                preds = get_preds(context)
                all_preds.append(preds)
                all_targets.append(targets)
                all_inputs.append(context[:, -1])
            
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            inputs = np.concatenate(all_inputs)
            
            metrics = compute_metrics(preds, targets)
            solver_preds = compiled_solver_batch(inputs, targets)
            solver_metrics = compute_metrics(solver_preds, targets)
            
            ood_results[f"objects_{n_obj_range[0]}-{n_obj_range[1]}"] = {
                'model': metrics, 'compiled_solver': solver_metrics
            }
            print(f"  {n_obj_range[0]}-{n_obj_range[1]}: exact={metrics['exact_match']:.3f}, "
                  f"fg={metrics['foreground_acc']:.3f}")
        
        return ood_results
    
    # =========================================================================
    # Phase 2: Probe Training (dxdy recovery from frozen backbone)
    # =========================================================================
    
    class DxDyProbe(nn.Module):
        """
        Lightweight probe to extract (dx, dy) from frozen backbone representations.
        
        Takes the last-layer hidden states, pools them, and predicts the translation.
        This tests: "Is the rule recoverable from the learned representation?"
        """
        def __init__(self, d_model=128):
            super().__init__()
            self.probe = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 14),  # 7 classes for dx + 7 for dy
            )
        
        def forward(self, h_pooled):
            return self.probe(h_pooled)
    
    def get_backbone_features(model, context):
        """
        Extract last-layer features from a trained model.
        Returns pooled representation for dxdy prediction.
        """
        batch, n_ctx, H, W = context.shape
        seq_len = H * W
        
        # Embed (same as model forward)
        context_flat = context.view(batch, n_ctx, seq_len)
        color_emb = model.color_embed(context_flat)
        
        grid_types = torch.arange(n_ctx, device=context.device)
        grid_type_emb = model.grid_type_embed(grid_types)
        grid_type_emb = grid_type_emb[None, :, None, :].expand(batch, -1, seq_len, -1)
        
        combined = torch.cat([color_emb, grid_type_emb], dim=-1)
        combined = combined.permute(0, 2, 1, 3).reshape(batch, seq_len, n_ctx * model.d_model)
        h = model.input_proj(combined)
        
        # Run through transformer blocks
        for block in model.blocks:
            h_norm = block['ln1'](h)
            rel_bias = block['rel_pos'](H, W, context.device)
            
            Q = h_norm @ block['attn'].in_proj_weight[:model.d_model].T + block['attn'].in_proj_bias[:model.d_model]
            K = h_norm @ block['attn'].in_proj_weight[model.d_model:2*model.d_model].T + block['attn'].in_proj_bias[model.d_model:2*model.d_model]
            V = h_norm @ block['attn'].in_proj_weight[2*model.d_model:].T + block['attn'].in_proj_bias[2*model.d_model:]
            
            d_head = model.d_model // model.n_heads
            Q = Q.view(batch, seq_len, model.n_heads, d_head).transpose(1, 2)
            K = K.view(batch, seq_len, model.n_heads, d_head).transpose(1, 2)
            V = V.view(batch, seq_len, model.n_heads, d_head).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) / (d_head ** 0.5)
            scores = scores + rel_bias[None, :, :, :]
            
            attn = F.softmax(scores, dim=-1)
            out = (attn @ V).transpose(1, 2).reshape(batch, seq_len, model.d_model)
            out = out @ block['attn'].out_proj.weight.T + block['attn'].out_proj.bias
            
            h = h + out
            h = h + block['mlp'](block['ln2'](h))
        
        # Pool over sequence for global representation
        h_pooled = h.mean(dim=1)  # (batch, d_model)
        return h_pooled
    
    def train_dxdy_probe(backbone_model, model_name, probe_epochs, steps_per_epoch,
                         batch_size, grid_size, n_objects_range, n_train_pairs, rng):
        """
        Phase 2: Train a probe to recover (dx, dy) from frozen backbone.
        
        This tests whether the rule is extractable from the learned representation.
        """
        print(f"\n--- Phase 2: Training dxdy probe for {model_name} ---")
        
        # Freeze backbone
        backbone_model.eval()
        for param in backbone_model.parameters():
            param.requires_grad = False
        
        # Create probe
        probe = DxDyProbe(d_model=128).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        
        best_acc = 0
        history = []
        
        for epoch in range(probe_epochs):
            probe.train()
            train_correct = 0
            train_total = 0
            
            for _ in range(steps_per_epoch):
                context, targets, translations = generate_batch_arc_style(
                    batch_size, grid_size, n_objects_range, n_train_pairs, rng
                )
                context = torch.from_numpy(context).to(device)
                
                # Get frozen backbone features
                with torch.no_grad():
                    h_pooled = get_backbone_features(backbone_model, context)
                
                # Train probe
                optimizer.zero_grad()
                dxdy_logits = probe(h_pooled)
                
                dx_targets = torch.tensor([t[0] + 3 for t in translations], device=device)
                dy_targets = torch.tensor([t[1] + 3 for t in translations], device=device)
                
                loss = (F.cross_entropy(dxdy_logits[:, :7], dx_targets) + 
                        F.cross_entropy(dxdy_logits[:, 7:], dy_targets)) / 2
                loss.backward()
                optimizer.step()
                
                dx_pred = dxdy_logits[:, :7].argmax(-1)
                dy_pred = dxdy_logits[:, 7:].argmax(-1)
                train_correct += ((dx_pred == dx_targets) & (dy_pred == dy_targets)).sum().item()
                train_total += len(translations)
            
            # Validation
            probe.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for _ in range(20):
                    context, targets, translations = generate_batch_arc_style(
                        batch_size, grid_size, n_objects_range, n_train_pairs, rng
                    )
                    context = torch.from_numpy(context).to(device)
                    h_pooled = get_backbone_features(backbone_model, context)
                    dxdy_logits = probe(h_pooled)
                    
                    dx_targets = torch.tensor([t[0] + 3 for t in translations], device=device)
                    dy_targets = torch.tensor([t[1] + 3 for t in translations], device=device)
                    dx_pred = dxdy_logits[:, :7].argmax(-1)
                    dy_pred = dxdy_logits[:, 7:].argmax(-1)
                    val_correct += ((dx_pred == dx_targets) & (dy_pred == dy_targets)).sum().item()
                    val_total += len(translations)
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            best_acc = max(best_acc, val_acc)
            
            history.append({'epoch': epoch + 1, 'train_acc': train_acc, 'val_acc': val_acc})
            print(f"  Probe epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        
        print(f"  Best probe accuracy: {best_acc:.3f}")
        
        # Test: use predicted dxdy + compiled solver, compare to model output
        print(f"\n  Testing: predicted_dxdy + compiled_solver vs model output")
        probe.eval()
        match_count = 0
        total_count = 0
        
        with torch.no_grad():
            for _ in range(20):
                context, targets, translations = generate_batch_arc_style(
                    batch_size, grid_size, n_objects_range, n_train_pairs, rng
                )
                context_t = torch.from_numpy(context).to(device)
                
                # Get model predictions
                model_preds = backbone_model(context_t).argmax(-1).cpu().numpy()
                
                # Get probe's dxdy predictions
                h_pooled = get_backbone_features(backbone_model, context_t)
                dxdy_logits = probe(h_pooled)
                dx_pred = (dxdy_logits[:, :7].argmax(-1) - 3).cpu().numpy()
                dy_pred = (dxdy_logits[:, 7:].argmax(-1) - 3).cpu().numpy()
                
                # Apply predicted dxdy with compiled solver
                test_inputs = context[:, -1]  # test input is last context grid
                for i in range(len(test_inputs)):
                    compiled_pred = apply_translation(test_inputs[i], int(dx_pred[i]), int(dy_pred[i]))
                    if np.array_equal(compiled_pred, model_preds[i]):
                        match_count += 1
                    total_count += 1
        
        match_rate = match_count / total_count
        print(f"  Match rate (probe_dxdy + solver == model): {match_rate:.3f}")
        
        return {
            'best_probe_acc': best_acc,
            'probe_history': history,
            'compiled_match_rate': match_rate,
        }
    
    # =========================================================================
    # Main Training Loop
    # =========================================================================
    
    n_train_pairs = 2
    n_context = n_train_pairs * 2 + 1
    n_memory = 8
    
    if quick:
        n_epochs = 15
        train_steps_per_epoch = 200
        val_steps = 20
        batch_size = 32
        probe_epochs = 10
    else:
        n_epochs = 30
        train_steps_per_epoch = 500
        val_steps = 50
        batch_size = 64
        probe_epochs = 20
    
    grid_size = 16
    n_objects_range = (1, 4)
    lr = 3e-4
    rng = np.random.default_rng(42)
    
    all_results = {}
    
    # Models to compare: (name, class, kwargs, use_taps)
    models_to_train = [
        ("baseline", ARCTransformerConditioned, {"n_context": n_context}, False),
        ("memory_8", ARCTransformerWithMemory, {"n_context": n_context, "n_memory": n_memory}, False),
    ]
    
    for model_name, ModelClass, model_kwargs, use_taps in models_to_train:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Training {model_name.upper()} (grid loss only)")
        print(f"{'='*60}")
        
        model = ModelClass(**model_kwargs).to(device)
        model = torch.compile(model)
        
        train_results = train_and_eval_model(
            model, model_name, n_epochs, train_steps_per_epoch,
            val_steps, batch_size, grid_size, n_objects_range,
            n_train_pairs, lr, rng, use_taps=use_taps
        )
        
        ood_results = eval_model_ood(
            model, model_name, val_steps, batch_size, grid_size,
            n_objects_range, n_train_pairs, rng
        )
        
        # Phase 2: Train dxdy probe on frozen backbone
        print(f"\n{'='*60}")
        print(f"PHASE 2: Probing {model_name.upper()} for dxdy recovery")
        print(f"{'='*60}")
        
        # Reload best checkpoint for probing
        checkpoint_path = f"/outputs/best_{model_name}.pt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = f"/outputs/last_{model_name}.pt"
        model.load_state_dict(torch.load(checkpoint_path))
        
        probe_results = train_dxdy_probe(
            model, model_name, probe_epochs, train_steps_per_epoch // 2,
            batch_size, grid_size, n_objects_range, n_train_pairs, rng
        )
        
        all_results[model_name] = {
            'training': train_results,
            'ood_evaluation': ood_results,
            'probe': probe_results,
        }
    
    # =========================================================================
    # Summary Comparison
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print("\n--- Phase 1: ID Performance (16x16, 1-4 objects) ---")
    for name in all_results:
        fg = all_results[name]['training']['best_fg_acc']
        print(f"  {name}: best_fg_acc = {fg:.3f}")
    
    print("\n--- Phase 1: Object Count OOD (5-7 objects) ---")
    for name in all_results:
        fg = all_results[name]['ood_evaluation']['objects_5-7']['model']['foreground_acc']
        print(f"  {name}: fg_acc = {fg:.3f}")
    
    print("\n--- Phase 2: dxdy Probe Recovery ---")
    for name in all_results:
        probe_acc = all_results[name]['probe']['best_probe_acc']
        match_rate = all_results[name]['probe']['compiled_match_rate']
        print(f"  {name}: probe_acc = {probe_acc:.3f}, compiled_match = {match_rate:.3f}")
    
    # Save results
    output_dir = Path("/outputs")
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    volume.commit()
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    return all_results


@app.local_entrypoint()
def main(quick: bool = False):
    import json
    print("Dispatching to Modal cloud...")
    results = train_experiment.remote(quick=quick)
    print("\nResults:")
    print(json.dumps(results, indent=2))
    
    with open("arc_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to arc_results.json")


if __name__ == "__main__":
    print("Run with: modal run train_modal.py --quick")
