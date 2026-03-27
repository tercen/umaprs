# TurboQuant — Experimental Vector Quantization for kNN

**Status: Experimental.** TurboQuant-based kNN methods are available as opt-in options. They are not used by default. Quality depends heavily on data dimensionality.

## What it does

TurboQuant compresses input vectors using (b-1)-bit MSE quantization + 1-bit QJL sign correction per coordinate. The total bit budget is b bits per coordinate with no overhead. Based on Google's TurboQuant paper (ICLR 2026).

## Algorithm (TurboQuant_prod)

1. **Normalize** each vector to unit length (store norms separately)
2. **Stage 1 — Rotation**: Random sign flips + Walsh-Hadamard transform (randomized orthogonal rotation). Decorrelates coordinates so each one is ~N(0,1).
3. **Stage 1 — MSE Quantize**: (b-1)-bit Lloyd-Max scalar quantizer per coordinate. Compute residual per coordinate.
4. **Stage 2 — QJL**: Apply a second (independent) randomized Hadamard to the residual vector. Store sign of each projected coordinate as 1 bit. This gives an unbiased inner product correction.
5. **Pack**: Each coordinate = (b-1) MSE bits + 1 QJL sign bit = b total bits.
6. **Distance**: `dot ≈ mse_dot + (π/2)/d² · ||r_i||·||r_j|| · sign_agreement`

The QJL sign agreement is computed via XOR + popcount — essentially free.

## Compression ratio

The compression depends only on b (total bits per coordinate), not on d:

| Total bits (b) | MSE bits | QJL bit | vs f64 (64-bit) | vs f32 (32-bit) |
|---|---|---|---|---|
| 4 | 3 | 1 | **16x** | **8x** |
| 5 | 4 | 1 | **12.8x** | **6.4x** |
| 6 | 5 | 1 | **10.7x** | **5.3x** |
| 8 | 7 | 1 | **8x** | **4x** |

## Quality depends on dimension AND bits

The QJL correction uses a constant approximation for `||r||` (residual norm). This relies on concentration-of-measure which improves with dimension d but degrades with fewer MSE bits.

| d | MSE bits | Total | vs f64 | vs f32 | ||r|| approx | QJL value |
|---|---|---|---|---|---|---|
| **128** | **3** | **4** | **16x** | **8x** | Good (~9%) | High (big residuals to correct) |
| 128 | 5 | 6 | 10.7x | 5.3x | Excellent (~3%) | Medium |
| 128 | 7 | 8 | 8x | 4x | Excellent (~2%) | Low (tiny residuals) |
| **64** | **3** | **4** | **16x** | **8x** | Good (~9%) | High |
| 64 | 4 | 5 | 12.8x | 6.4x | Good (~7%) | Medium-high |
| 64 | 5 | 6 | 10.7x | 5.3x | Very good (~5%) | Medium |
| 64 | 7 | 8 | 8x | 4x | Very good (~3%) | Low |
| **32** | **3** | **4** | **16x** | **8x** | Marginal (~18%) | High but noisy |
| 32 | 4 | 5 | 12.8x | 6.4x | OK (~14%) | Medium-high |
| 32 | 5 | 6 | 10.7x | 5.3x | Good (~10%) | Medium |
| 32 | 7 | 8 | 8x | 4x | Good (~6%) | Low |
| **16** | **3** | **4** | **16x** | **8x** | Poor (~25%) | Noisy |
| 16 | 5 | 6 | 10.7x | 5.3x | OK (~14%) | Medium |
| 16 | 7 | 8 | 8x | 4x | Marginal (~18%) | Low |

**Reading the table**: "||r|| approx" is the relative variance of the constant residual norm approximation — lower is better. "QJL value" is how much the sign correction helps — high means the MSE alone is poor and QJL adds significant information.

## Recommended bit budget per dimension

| d | Optimal total bits | Compression | Why |
|---|---|---|---|
| ≥ 128 | 4-bit (3+1) | 16x | Concentration is strong, QJL correction is accurate |
| 64 | 4-5 bit | 12-16x | 4-bit works well, 5-bit is safer |
| 32 | 5-6 bit | 10-13x | 4-bit is marginal, 5-bit balances compression and quality |
| 16 | 6-8 bit | 8-11x | Need more MSE bits to compensate for weak concentration |
| ≤ 8 | Skip TQ | — | Concentration too weak, just use f32 |

## When to use

TurboQuant is useful when:
- Memory is constrained (e.g., very large datasets that don't fit in RAM)
- Dimensions are high enough for concentration-of-measure (>= 16 dims, ideally >= 32)
- Approximate kNN is acceptable

## When NOT to use

- **Dimensions < 16** — concentration too weak, quantization distances are unreliable
- **When exact kNN is critical** — use kd-tree (dims ≤ 40) or brute-force
- **Small datasets** — brute-force is fast enough and exact

## Usage

```rust
use umaprs::{UMAP, KnnMethod};

// TQ 4-bit brute-force (3-bit MSE + 1-bit QJL, 16x compression)
let embedding = UMAP::new()
    .knn_method(KnnMethod::TurboQuant4)
    .fit_transform(&data);

// TQ 8-bit brute-force (7-bit MSE + 1-bit QJL, 8x compression)
let embedding = UMAP::new()
    .knn_method(KnnMethod::TurboQuant8)
    .fit_transform(&data);

// TQ 4-bit + HNSW (approximate kNN on quantized distances)
let embedding = UMAP::new()
    .knn_method(KnnMethod::TurboQuant4Hnsw)
    .fit_transform(&data);

// GPU TQ4 (CUDA kernel with shared memory codebook + QJL correction)
let embedding = UMAP::new()
    .knn_method(KnnMethod::GpuTQ4)
    .fit_transform(&data);
```

## Available methods

| KnnMethod | Bits | kNN search | Distance function | Quality |
|---|---|---|---|---|
| `TurboQuant4` | 4 (3+1) | Brute-force | MSE + QJL (approx_dist_sq) | Best for d ≥ 64 |
| `TurboQuant8` | 8 (7+1) | Brute-force | MSE + QJL (approx_dist_sq) | Best for d = 16-64 |
| `TurboQuant4Hnsw` | 4 (3+1) | HNSW | MSE + QJL (approx_dist_sq) | Faster, lower quality |
| `TurboQuant8Hnsw` | 8 (7+1) | HNSW | MSE + QJL (approx_dist_sq) | Faster, moderate quality |
| `GpuTQ4` | 4 (3+1) | GPU brute-force | CUDA kernel + QJL | Best for large data + GPU |

All methods refine their top-2k candidates with exact f64 distances on the original data.

## Implementation details

### QJL projection
The QJL uses a second randomized Hadamard transform (independent from Stage 1) as the random projection S. This is orthogonal (S·S^T = I exactly), so `<q_x, q_y>` via XOR+popcount gives the exact projected inner product — no approximation.

### Constant ||r|| approximation
The residual norm `||r||` is approximated as a global constant `√(d · MSE_per_coord)` rather than stored per-point. This relies on concentration-of-measure: after the Hadamard rotation, all coordinates are ~N(0,1), so quantization MSE is uniform across points. The approximation quality depends on d and the MSE bit budget (see table above).

### Codebook
3-bit (8-level) Lloyd-Max codebook optimized for N(0,1) distribution. Precomputed centroids and decision boundaries. For 7-bit mode, a uniform quantizer over [-3.5, 3.5] is used (Lloyd-Max and uniform converge at high bit counts).

## References

- TurboQuant paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- QJL (Quantized Johnson-Lindenstrauss) for unbiased inner product estimation
- Walsh-Hadamard transform for fast orthogonal random rotation
- Lloyd-Max scalar quantization for optimal codebook design
