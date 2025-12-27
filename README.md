# ARC Primitive Translation Experiment

Testing whether a transformer can learn a simple ARC-like primitive: **translate all objects by a fixed (dx, dy)**.

## Goal

We are NOT trying to solve ARC. We're testing:
1. Can a transformer learn a single ARC-like primitive?
2. Is the computation recoverable (can we identify object-centric translation from behavior)?
3. Is it distillable into an explicit object-slot architecture? (Phase 2)

## Design Choices

### Data
- **1 cell = 1 token** (no patches, to preserve single-cell flags)
- Colors 0-9 (ARC palette), 0 = background
- Objects: random 4-connected polyomino blobs
- ~30% of tasks include single-cell "flag" markers
- ~20% include thin 1-cell-wide lines
- Objects have ≥1 cell gap between them
- Translation: (dx, dy) sampled from [-3, 3] \ {(0,0)}, in-bounds

### Model
- Transformer over flattened cells (16×16 = 256 tokens)
- Color embedding, NO absolute position embeddings
- 2D relative position bias (row + col offsets)
- Small: d_model=128, 4 heads, 4 layers

### Evaluation
- In-distribution: 16×16 grid, 1-4 objects
- OOD grid size: 24×24, 32×32
- OOD object count: 5-7, 8-10

## Files

- `data_gen.py` - Synthetic ARC task generator
- `models.py` - NumPy baseline model (for reference)
- `train_modal.py` - PyTorch training on Modal GPU
- `eval.py` - Evaluation and probes (Phase 2)

## Usage

```bash
# Quick test (~2-3 min on A10)
modal run train_modal.py --quick

# Full training (~10-15 min)
modal run train_modal.py
```

## Phase 1 Status

- [x] Data generator with polyominos, flags, lines
- [x] 2D transformer with relative position bias
- [x] Modal training script
- [x] OOD evaluation (grid size, object count)

## Phase 2 (TODO)

- [ ] Slot attention model
- [ ] Behavioral probes (color permutation equivariance, translation commutation)
- [ ] Blind packet for Gemini evaluation
