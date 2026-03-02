# Ultra-Light Grid Sweep Analysis

## Configuration Summary

**Method**: Grid search (exhaustive)
**Target**: Small, efficient transformer architectures for HMM tasks

## Parameter Space

### Architecture Parameters (Variable):
- `n_layer`: [1, 2] → 2 values
- `n_embd`: [32, 64] → 2 values
- `n_head`: [2, 4] → 2 values
- `d_head`: [8, 16] → 2 values
- `n_inner`: [64, 128] → 2 values
- `n_ctx`: [6, 10] → 2 values

### Fixed Parameters:
- `attn_only`: false (always include FFN)
- `normalization_type`: "LN" (always use LayerNorm)
- `learning_rate`: 0.0003
- `batch_size`: 128
- `scheduler_type`: "warmup_cosine"

## Grid Size Calculation

### Raw Combinations:
2 (n_layer) × 2 (n_embd) × 2 (n_head) × 2 (d_head) × 2 (n_inner) × 2 (n_ctx) = **128 runs**

### Valid Combinations (after constraint filtering):

**Constraint**: `n_embd = n_head × d_head`

Valid architecture combinations:
1. n_embd=32, n_head=2, d_head=16 ✅
2. n_embd=32, n_head=4, d_head=8 ✅
3. n_embd=64, n_head=4, d_head=16 ✅
4. n_embd=64, n_head=2, d_head=32 ❌ (d_head=32 not in grid)

**Valid combinations**: 3 out of 8

Invalid combinations (will be rejected by train.py):
- n_embd=32, n_head=2, d_head=8 → 32 ≠ 2×8=16
- n_embd=32, n_head=4, d_head=16 → 32 ≠ 4×16=64
- n_embd=64, n_head=2, d_head=8 → 64 ≠ 2×8=16
- n_embd=64, n_head=2, d_head=16 → 64 ≠ 2×16=32
- n_embd=64, n_head=4, d_head=8 → 64 ≠ 4×8=32

### Expected Grid Size:
**48 valid runs** = 3 (valid arch) × 2 (n_layer) × 2 (n_inner) × 2 (n_ctx) × 1 (fixed params)

### Invalid Run Handling:
- Invalid configurations will fail validation in `train.py` line 310-322
- These runs will be logged to WandB with `config_valid: False`
- Only **48 runs** will actually train
- **80 runs** will be rejected early (no training cost)

## Model Sizes (Valid Architectures)

| n_layer | n_embd | n_head | d_head | n_inner | n_ctx | Approx Params |
|---------|--------|--------|--------|---------|-------|---------------|
| 1       | 32     | 2      | 16     | 64      | 6     | ~3K           |
| 1       | 32     | 2      | 16     | 128     | 6     | ~5K           |
| 1       | 32     | 4      | 8      | 64      | 6     | ~3K           |
| 1       | 32     | 4      | 8      | 128     | 6     | ~5K           |
| 1       | 64     | 4      | 16     | 64      | 6     | ~11K          |
| 1       | 64     | 4      | 16     | 128     | 6     | ~17K          |
| 2       | 32     | 2      | 16     | 64      | 6     | ~6K           |
| 2       | 32     | 2      | 16     | 128     | 6     | ~10K          |
| 2       | 32     | 4      | 8      | 64      | 6     | ~6K           |
| 2       | 32     | 4      | 8      | 128     | 6     | ~10K          |
| 2       | 64     | 4      | 16     | 64      | 6     | ~22K          |
| 2       | 64     | 4      | 16     | 128     | 6     | ~34K          |

(Same architectures repeated for n_ctx=10)

All models are **very small** (< 50K parameters), perfect for efficient exploration.

## Training Time Estimate

Per run (100,000 steps):
- With GPU: ~2-4 hours per run
- 48 valid runs: ~100-200 GPU hours total
- With 4 GPUs: ~25-50 hours wall time
- Invalid runs: <1 minute each (early rejection)

## Research Questions This Grid Addresses

1. **Depth**: Does 1 layer vs 2 layers matter?
2. **Width**: 32 vs 64 embedding dimension impact?
3. **Attention heads**: 2 vs 4 heads (with matched d_head)?
4. **FFN size**: 64 vs 128 inner dimension?
5. **Context length**: How does performance scale from ctx=6 to ctx=10?

## Usage

Launch the grid sweep:
```bash
python launch_sweep.py --config sweep_config_grid.yaml --num_agents 1
```

Or manually:
```bash
wandb sweep sweep_config_grid.yaml
wandb agent <entity>/hidden-markov-model/<sweep-id>
```

For multiple GPUs:
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep-id>

# Terminal 2
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep-id>

# etc...
```
