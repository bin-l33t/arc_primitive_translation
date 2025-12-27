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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # =========================================================================
    # Data Generation (copied from data_gen.py for Modal)
    # =========================================================================
    
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
        
        # Main objects
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
    
    def sample_valid_translation(grid, objects, max_t, rng):
        H, W = grid.shape
        all_cells = []
        for obj in objects:
            r0, c0 = obj['top_left']
            for dr, dc in obj['cells']:
                all_cells.append((r0 + dr, c0 + dc))
        
        if not all_cells:
            return None
        
        min_r, max_r = min(c[0] for c in all_cells), max(c[0] for c in all_cells)
        min_c, max_c = min(c[1] for c in all_cells), max(c[1] for c in all_cells)
        
        dx_min, dx_max = max(-max_t, -min_r), min(max_t, H - 1 - max_r)
        dy_min, dy_max = max(-max_t, -min_c), min(max_t, W - 1 - max_c)
        
        valid = [(dx, dy) for dx in range(dx_min, dx_max + 1) 
                 for dy in range(dy_min, dy_max + 1) if dx != 0 or dy != 0]
        
        return valid[rng.integers(len(valid))] if valid else None
    
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
    
    def generate_batch(batch_size, grid_size, n_objects_range, rng):
        inputs = np.zeros((batch_size, grid_size, grid_size), dtype=np.int64)
        outputs = np.zeros((batch_size, grid_size, grid_size), dtype=np.int64)
        
        for i in range(batch_size):
            n_objects = rng.integers(n_objects_range[0], n_objects_range[1] + 1)
            for _ in range(10):
                input_grid, objects = generate_scene(grid_size, n_objects, rng)
                if not objects:
                    continue
                translation = sample_valid_translation(input_grid, objects, 3, rng)
                if translation:
                    dx, dy = translation
                    output_grid = translate_scene(input_grid, objects, dx, dy)
                    inputs[i] = input_grid
                    outputs[i] = output_grid
                    break
        
        return inputs, outputs
    
    # =========================================================================
    # Model (PyTorch version)
    # =========================================================================
    
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
    
    class ARCTransformer(nn.Module):
        def __init__(self, n_colors=10, d_model=128, n_heads=4, n_layers=4, d_ff=256, max_offset=8):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            
            self.embedding = nn.Embedding(n_colors, d_model)
            
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
        
        def forward(self, x):
            batch, H, W = x.shape
            seq_len = H * W
            
            h = self.embedding(x.view(batch, -1))
            
            for block in self.blocks:
                h_norm = block['ln1'](h)
                
                # Custom attention with relative position bias
                rel_bias = block['rel_pos'](H, W, x.device)
                
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
    
    # =========================================================================
    # Training
    # =========================================================================
    
    # Config
    if quick:
        n_epochs = 5
        train_steps_per_epoch = 100
        val_steps = 20
        batch_size = 32
    else:
        n_epochs = 20
        train_steps_per_epoch = 500
        val_steps = 50
        batch_size = 64
    
    grid_size = 16
    n_objects_range = (1, 4)
    lr = 3e-4
    
    print(f"\n{'='*60}")
    print("Training ARC Translation Model")
    print(f"{'='*60}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Objects: {n_objects_range}")
    print(f"Epochs: {n_epochs}, Steps/epoch: {train_steps_per_epoch}")
    
    # Model
    model = ARCTransformer().to(device)
    model = torch.compile(model)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs * train_steps_per_epoch)
    
    rng = np.random.default_rng(42)
    history = []
    best_acc = 0
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for step in range(train_steps_per_epoch):
            inputs, targets = generate_batch(batch_size, grid_size, n_objects_range, rng)
            inputs = torch.from_numpy(inputs).to(device)
            targets = torch.from_numpy(targets).to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, 10), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_correct += (logits.argmax(-1) == targets).sum().item()
            train_total += targets.numel()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = generate_batch(batch_size, grid_size, n_objects_range, rng)
                inputs = torch.from_numpy(inputs).to(device)
                targets = torch.from_numpy(targets).to(device)
                
                logits = model(inputs)
                val_correct += (logits.argmax(-1) == targets).sum().item()
                val_total += targets.numel()
        
        train_loss /= train_steps_per_epoch
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        })
        
        print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "/outputs/best_model.pt")
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val accuracy: {best_acc:.4f}")
    
    # =========================================================================
    # OOD Evaluation
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("OOD Evaluation")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load("/outputs/best_model.pt"))
    model.eval()
    
    ood_results = {}
    
    # OOD grid sizes
    for test_size in [16, 24, 32]:
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = generate_batch(batch_size, test_size, n_objects_range, rng)
                inputs = torch.from_numpy(inputs).to(device)
                targets = torch.from_numpy(targets).to(device)
                
                logits = model(inputs)
                correct += (logits.argmax(-1) == targets).sum().item()
                total += targets.numel()
        
        acc = correct / total
        ood_results[f"grid_{test_size}x{test_size}"] = acc
        print(f"Grid {test_size}x{test_size}: {acc:.4f}")
    
    # OOD object counts
    for n_obj_range in [(1, 4), (5, 7), (8, 10)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = generate_batch(batch_size, grid_size, n_obj_range, rng)
                inputs = torch.from_numpy(inputs).to(device)
                targets = torch.from_numpy(targets).to(device)
                
                logits = model(inputs)
                correct += (logits.argmax(-1) == targets).sum().item()
                total += targets.numel()
        
        acc = correct / total
        ood_results[f"objects_{n_obj_range[0]}-{n_obj_range[1]}"] = acc
        print(f"Objects {n_obj_range[0]}-{n_obj_range[1]}: {acc:.4f}")
    
    # Save results
    results = {
        "training": {
            "best_val_acc": best_acc,
            "epochs": n_epochs,
            "grid_size": grid_size,
            "n_objects_range": n_objects_range,
        },
        "ood_evaluation": ood_results,
        "history": history,
    }
    
    output_dir = Path("/outputs")
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    volume.commit()
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    return results


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
