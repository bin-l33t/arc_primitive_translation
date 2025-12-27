"""
2D Transformer for ARC primitive learning.

Key design choices:
- 1 cell = 1 token (no patches, to preserve single-cell flags)
- Color embedding for each cell
- NO absolute position embeddings
- 2D relative position bias (row/col offsets)
- Small model: d_model=128, 4 heads, 4 layers
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for 2D transformer."""
    n_colors: int = 10  # 0-9 color palette
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    max_grid_size: int = 32  # for relative position bias range
    rel_pos_max: int = 8  # clip relative positions to [-max, max]
    dropout: float = 0.0


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class Embedding:
    """Color embedding layer."""
    
    def __init__(self, n_colors: int, d_model: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / (n_colors + d_model))
        self.W = rng.normal(0, scale, (n_colors, d_model)).astype(np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, H, W) int -> (batch, H*W, d_model)"""
        batch, H, W = x.shape
        x_flat = x.reshape(batch, -1)  # (batch, H*W)
        return self.W[x_flat]  # (batch, H*W, d_model)
    
    def params(self) -> List[np.ndarray]:
        return [self.W]


class RelativePositionBias2D:
    """
    2D relative position bias for attention.
    
    For each pair of positions (i, j), compute relative row and column offsets,
    and look up a learned bias value.
    """
    
    def __init__(self, n_heads: int, max_offset: int, rng: np.random.Generator):
        self.n_heads = n_heads
        self.max_offset = max_offset
        
        # Separate biases for row and column offsets
        # Range: [-max_offset, max_offset] -> 2*max_offset + 1 values
        n_buckets = 2 * max_offset + 1
        
        # Shape: (n_heads, n_buckets) for row, (n_heads, n_buckets) for col
        self.row_bias = rng.normal(0, 0.02, (n_heads, n_buckets)).astype(np.float32)
        self.col_bias = rng.normal(0, 0.02, (n_heads, n_buckets)).astype(np.float32)
    
    def get_bias(self, H: int, W: int) -> np.ndarray:
        """
        Compute relative position bias matrix.
        
        Returns: (n_heads, H*W, H*W) bias values to add to attention scores.
        """
        seq_len = H * W
        
        # Create position grids
        positions = np.arange(seq_len)
        rows = positions // W  # (seq_len,)
        cols = positions % W   # (seq_len,)
        
        # Compute pairwise relative offsets
        # rel_rows[i, j] = rows[i] - rows[j]
        rel_rows = rows[:, None] - rows[None, :]  # (seq_len, seq_len)
        rel_cols = cols[:, None] - cols[None, :]  # (seq_len, seq_len)
        
        # Clip to [-max_offset, max_offset] and shift to [0, 2*max_offset]
        rel_rows_clipped = np.clip(rel_rows, -self.max_offset, self.max_offset) + self.max_offset
        rel_cols_clipped = np.clip(rel_cols, -self.max_offset, self.max_offset) + self.max_offset
        
        # Look up biases: (n_heads, seq_len, seq_len)
        row_bias = self.row_bias[:, rel_rows_clipped]  # (n_heads, seq_len, seq_len)
        col_bias = self.col_bias[:, rel_cols_clipped]  # (n_heads, seq_len, seq_len)
        
        return row_bias + col_bias
    
    def params(self) -> List[np.ndarray]:
        return [self.row_bias, self.col_bias]


class MultiHeadAttention:
    """Multi-head self-attention with 2D relative position bias."""
    
    def __init__(self, d_model: int, n_heads: int, max_offset: int, rng: np.random.Generator):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        # QKV projections
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        
        # 2D relative position bias
        self.rel_pos = RelativePositionBias2D(n_heads, max_offset, rng)
    
    def forward(self, x: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        x: (batch, seq_len, d_model)
        H, W: grid dimensions (seq_len = H * W)
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = x @ self.W_q  # (batch, seq, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape to multi-head
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        # Now: (batch, n_heads, seq_len, d_head)
        
        # Attention scores
        scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale  # (batch, n_heads, seq, seq)
        
        # Add 2D relative position bias
        rel_bias = self.rel_pos.get_bias(H, W)  # (n_heads, seq, seq)
        scores = scores + rel_bias[None, :, :, :]
        
        # Softmax and weighted sum
        attn = softmax(scores, axis=-1)
        out = attn @ V  # (batch, n_heads, seq, d_head)
        
        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        
        # Output projection
        return out @ self.W_o
    
    def params(self) -> List[np.ndarray]:
        return [self.W_q, self.W_k, self.W_v, self.W_o] + self.rel_pos.params()


class MLP:
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.W1 = rng.normal(0, scale1, (d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
    
    def params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]


class TransformerBlock:
    """Pre-LN transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_offset: int, rng: np.random.Generator):
        self.attn = MultiHeadAttention(d_model, n_heads, max_offset, rng)
        self.mlp = MLP(d_model, d_ff, rng)
        
        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model, dtype=np.float32)
        self.ln1_beta = np.zeros(d_model, dtype=np.float32)
        self.ln2_gamma = np.ones(d_model, dtype=np.float32)
        self.ln2_beta = np.zeros(d_model, dtype=np.float32)
    
    def forward(self, x: np.ndarray, H: int, W: int) -> np.ndarray:
        # Pre-LN attention
        h = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        x = x + self.attn.forward(h, H, W)
        
        # Pre-LN MLP
        h = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        x = x + self.mlp.forward(h)
        
        return x
    
    def params(self) -> List[np.ndarray]:
        return ([self.ln1_gamma, self.ln1_beta, self.ln2_gamma, self.ln2_beta] + 
                self.attn.params() + self.mlp.params())


class ARCTransformer:
    """
    Full transformer for ARC translation task.
    
    Input: (batch, H, W) grid of color indices 0-9
    Output: (batch, H, W, n_colors) logits for each cell
    """
    
    def __init__(self, config: TransformerConfig, seed: int = 42):
        self.config = config
        rng = np.random.default_rng(seed)
        
        # Embedding
        self.embedding = Embedding(config.n_colors, config.d_model, rng)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.rel_pos_max, rng
            )
            for _ in range(config.n_layers)
        ]
        
        # Final layer norm and output projection
        self.ln_final_gamma = np.ones(config.d_model, dtype=np.float32)
        self.ln_final_beta = np.zeros(config.d_model, dtype=np.float32)
        
        scale = np.sqrt(2.0 / (config.d_model + config.n_colors))
        self.output_proj = rng.normal(0, scale, (config.d_model, config.n_colors)).astype(np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, H, W) int array of colors
        Returns: (batch, H, W, n_colors) logits
        """
        batch, H, W = x.shape
        seq_len = H * W
        
        # Embed colors
        h = self.embedding.forward(x)  # (batch, H*W, d_model)
        
        # Transformer blocks
        for block in self.blocks:
            h = block.forward(h, H, W)
        
        # Final layer norm
        h = layer_norm(h, self.ln_final_gamma, self.ln_final_beta)
        
        # Output projection
        logits = h @ self.output_proj  # (batch, H*W, n_colors)
        
        # Reshape to grid
        return logits.reshape(batch, H, W, self.config.n_colors)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted color indices."""
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)
    
    def params(self) -> List[np.ndarray]:
        params = self.embedding.params()
        for block in self.blocks:
            params.extend(block.params())
        params.extend([self.ln_final_gamma, self.ln_final_beta, self.output_proj])
        return params
    
    def count_params(self) -> int:
        return sum(p.size for p in self.params())


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Cross-entropy loss for grid prediction.
    
    logits: (batch, H, W, n_colors)
    targets: (batch, H, W) int
    
    Returns: (loss, d_logits)
    """
    batch, H, W, n_colors = logits.shape
    
    # Flatten
    logits_flat = logits.reshape(-1, n_colors)  # (batch*H*W, n_colors)
    targets_flat = targets.reshape(-1)  # (batch*H*W,)
    
    # Softmax
    probs = softmax(logits_flat, axis=-1)
    
    # Gather target probs
    n_samples = targets_flat.shape[0]
    target_probs = probs[np.arange(n_samples), targets_flat]
    
    # Loss
    loss = -np.mean(np.log(target_probs + 1e-10))
    
    # Gradient
    d_logits_flat = probs.copy()
    d_logits_flat[np.arange(n_samples), targets_flat] -= 1
    d_logits_flat /= n_samples
    
    d_logits = d_logits_flat.reshape(batch, H, W, n_colors)
    
    return loss, d_logits


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing ARC Transformer...")
    
    config = TransformerConfig()
    model = ARCTransformer(config, seed=42)
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Test forward pass
    x = np.random.randint(0, 10, (4, 16, 16), dtype=np.int32)
    logits = model.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test loss
    targets = np.random.randint(0, 10, (4, 16, 16), dtype=np.int32)
    loss, d_logits = cross_entropy_loss(logits, targets)
    print(f"Loss: {loss:.4f}")
    print(f"Gradient shape: {d_logits.shape}")
    
    # Test prediction
    preds = model.predict(x)
    print(f"Predictions shape: {preds.shape}")
    
    # Test with different grid size
    x_large = np.random.randint(0, 10, (2, 24, 24), dtype=np.int32)
    logits_large = model.forward(x_large)
    print(f"\nLarge grid (24x24):")
    print(f"  Input: {x_large.shape}, Output: {logits_large.shape}")
