# TurboQuant — Experimental Vector Quantization for kNN

**Status: Experimental.** TurboQuant-based kNN methods are available as opt-in options. They are not used by default. Quality depends heavily on data dimensionality.

## What it does

TurboQuant compresses input vectors from 64-bit floats to 4-bit or 8-bit per coordinate, reducing kNN memory by 11-16x. The compression uses a randomized Hadamard transform followed by Lloyd-Max scalar quantization, inspired by Google's TurboQuant paper (ICLR 2026).

## Algorithm

1. **Normalize** each vector to unit length (store norms separately)
2. **Random sign flips** + **Walsh-Hadamard transform** — decorrelates coordinates so each one is approximately independently distributed
3. **Scalar quantize** each coordinate using a precomputed Lloyd-Max codebook (4-bit: 16 levels, 8-bit: 256 levels)
4. **Distance computation** — approximate Euclidean distance from quantized representation using inner product in the rotated space

## When to use

TurboQuant is useful when:
- Memory is constrained (e.g., very large datasets that don't fit in RAM)
- Dimensions are high enough for concentration-of-measure guarantees (>= 64 dims)
- Approximate kNN is acceptable

## When NOT to use

- **Dimensions < 32** — quantization noise is too high relative to actual distances. At 32 dims, 4-bit quality is ~60% of exact kNN. Use kd-tree or plain HNSW instead.
- **When quality matters most** — kd-tree gives exact neighbors and is the better default for dims <= 40.
- **Small datasets** — brute-force is fast enough and exact.

## Quality by dimensionality

Results on real datasets (separation ratio vs uwot reference):

| Dims | 4-bit + kd-tree | 8-bit + kd-tree | kd-tree (exact) |
|------|-----------------|-----------------|-----------------|
| 5 (crabs) | 101% | 100% | 100% |
| 32 (CyTOF) | 106% | 102% | 103% |
| 64 (digits) | 103% | 86% | — |

At 32+ dims, 4-bit + kd-tree can actually *improve* quality through beneficial regularization of the neighbor graph. At lower dims, it degrades quality.

Note: 4-bit + HNSW performs poorly (60% at 32 dims) because HNSW approximate search compounds with quantization noise. Prefer kd-tree when using TurboQuant.

## Memory compression

| Dims | 4-bit compression | 8-bit compression |
|------|-------------------|-------------------|
| 32 | ~11x | ~6x |
| 50 | ~11x | ~6x |
| 64 | ~14x | ~7.5x |
| 200 | ~12x | ~6x |

## Usage

```rust
use umaprs::{UMAP, KnnMethod};

// 4-bit TurboQuant with kd-tree (best quality)
let embedding = UMAP::new()
    .knn_method(KnnMethod::TurboQuant4KdTree)
    .fit_transform(&data);

// 8-bit TurboQuant with kd-tree (more conservative)
let embedding = UMAP::new()
    .knn_method(KnnMethod::TurboQuant8KdTree)
    .fit_transform(&data);

// 4-bit TurboQuant with HNSW (fastest, lowest quality)
let embedding = UMAP::new()
    .knn_method(KnnMethod::TurboQuant4Hnsw)
    .fit_transform(&data);
```

## Available methods

| KnnMethod | Quantization | Index | Exact? | Speed | Quality |
|-----------|-------------|-------|--------|-------|---------|
| `TurboQuant4KdTree` | 4-bit | kd-tree | Exact on quantized | Medium | Best |
| `TurboQuant8KdTree` | 8-bit | kd-tree | Exact on quantized | Medium | Good |
| `TurboQuant4Hnsw` | 4-bit | HNSW | Approximate | Fast | Low |
| `TurboQuant8Hnsw` | 8-bit | HNSW | Approximate | Fast | Medium |

All TurboQuant methods refine their top-2k candidates with exact f64 distances on the original data to mitigate quantization error.

## References

- Google TurboQuant paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- Walsh-Hadamard transform for random rotation
- Lloyd-Max scalar quantization for optimal codebook design
