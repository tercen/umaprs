# umaprs

A fast Rust implementation of UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.

UMAP as `umap-learn` defines it — every stage checked against `umap-learn 0.5.12` on the same kNN — with a transform that projects new points the way `umap-learn`'s does, seeded, and parallel. Minimal dependencies, self-contained binary (~1.5 MB). See **Parity** and **Envelope** below for what is measured, and `STATUS.md` for how.

## Performance

Measured on the same deterministic 50k × 32 fit (`examples/time_fit.rs`), 16 cores:

| | this branch | base commit |
|---|---|---|
| fit, 200 epochs | **5.96 s** | 10.01 s |

The branch replaced an approximate `pow` (5.3% worst error) with an exact one *and* got faster:
the fuzzy set went from a serial `HashMap` to a parallel CSR merge (8.86 s), then the HNSW index
build went parallel (5.96 s). The transform of 100k points against a 100k training set at 40 dims
takes **9.8 s** on the HNSW path (kNN + σ + init 7.8 s, SGD 2.0 s); the same on the old kd-tree
cutoff took 139.7 s, and the brute-force scan before that would have taken an hour at cohort
scale. Reference engines on the same 50k × 32
data, one fit: `umap-learn 0.5.12` 76.8 s, `uwot 0.2.5` (`n_sgd_threads = 0`) ~20 s.

## Usage

```rust
use umaprs::UMAP;

let embedding = UMAP::new()
    .n_neighbors(15)
    .min_dist(0.1)
    .n_epochs(200)
    .random_state(42)
    .fit_transform(&data);
```

## Parameters

```rust
UMAP::new()
    .n_neighbors(15)             // kNN graph size (default: 15)
    .n_components(2)             // output dimensions (default: 2)
    .min_dist(0.1)               // min distance in embedding (default: 0.1)
    .spread(1.0)                 // embedding spread (default: 1.0)
    .learning_rate(1.0)          // SGD step size (default: 1.0)
    .n_epochs(200)               // optimization epochs, 0=auto (default: 0)
    .negative_sample_rate(5.0)   // repulsive samples per edge (default: 5)
    .repulsion_strength(1.0)     // repulsion force multiplier (default: 1.0)
    .init(InitMethod::Auto)      // Auto | Spectral | Pca | Random
    .knn_method(KnnMethod::Auto) // Auto | KdTree | Hnsw | BruteForce | TurboQuant*
    .pca(50)                     // optional PCA dim reduction before kNN
    .train_size(0.1)             // fit on subset, transform rest (for large data)
    .random_state(42)            // seed: kNN (HNSW), init noise, SGD sampling, transform
    .threads(0)                  // rayon pool size; 0 = rayon's choice (honours a CPU quota)
```

## kNN Methods

| Method | Type | Best for |
|---|---|---|
| `KdTree` | Exact | dims <= 16 (default for this range; at 40 dims a kd-tree barely prunes — measured, see `STATUS.md`) |
| `Hnsw` | Approximate, seeded from `random_state`, 2k exact refine | dims > 16 (default) |
| `BruteForce` | Exact | n <= 500 (default for this range) |
| `TurboQuant4KdTree` | Approximate | Memory-constrained, high-dim (experimental) |
| `TurboQuant8KdTree` | Approximate | Memory-constrained, moderate-dim (experimental) |

See [docs/turboquant.md](docs/turboquant.md) for details on TurboQuant methods.

## Fit / Transform

```rust
let (embedding, model) = UMAP::new().n_neighbors(15).min_dist(0.01).random_state(42).fit(&train);
let projected = model.transform(&rest);   // umap-learn's transform, not a placement
```

`transform` is `umap-learn`'s, stage for stage: each new point's neighbours among the training
points (kd-tree or HNSW — never a scan), its own σ with ρ = 0 (`local_connectivity − 1`, the
reference's convention for a query), bipartite memberships, the membership-weighted mean of the
neighbours' positions as the start, then negative-sampled SGD against the **fixed** training
map at a quarter of the learning rate for 30 / 100 epochs (or a third of the fit's). Only the
new point moves, so every point is independent: the loop is parallel with one seeded RNG per
point (`model.transform_seed`, default 42 as upstream) and bit-identical at any thread count.

`.train_size(f)` still fits on a random fraction and transforms the rest; a per-group draw
(so many cells per sample) is the caller's to make with `fit` + `transform`.

## Threads and reproducibility

`.threads(n)` sizes the rayon pool for kNN, the fuzzy set, the SGD and the transform; `0` (the
default) leaves rayon its choice, which honours a container's CPU quota. Every stage is
deterministic by construction or seeded per unit of work except the fit's HogWild SGD, whose
float summation order depends on the thread count: **`threads(1)` is bit-repeatable; any other
count is repeatable up to that order** — the same contract as `umap-learn`'s parallel mode.

## Parity

`tests/umap_learn_parity.rs` holds each stage to `umap-learn 0.5.12` given the same kNN:
σ and ρ, the memberships, the symmetric graph, `a`/`b`, and the transform's kNN, σ,
memberships and initial positions. Two tolerances: **1e-12** against a line-for-line float64
transcription of the reference's functions (the arithmetic is the same), **1e-4** against a real
run (its stage functions are numba-compiled for float32). Fixtures are synthetic, written at
17 significant digits by `fixtures/gen_umap_learn_fixtures.py`.

## Envelope

`scripts/envelope.py` compares engines at three seeds on the metrics the cytometry parameter sweep used — kNN
purity against labels, trustworthiness, seed-to-seed neighbour overlap. The table is in
`STATUS.md`.

## Data Preprocessing

For CyTOF / mass cytometry data, always apply arcsinh transformation before UMAP:

```r
# R
data_transformed <- asinh(data / 5)  # cofactor 5
```

Drop housekeeping markers (Time, Cell_length, DNA, Viability) and keep only protein markers.

## Building

```bash
cargo build --release
```

OpenBLAS is compiled from source and statically linked (first build takes ~60s).

## Examples

```bash
# Crabs dataset (200 samples, 5 dims)
cargo run --release --example crabs

# Performance benchmark
cargo run --release --example bench

# Cytometry comparison (requires data download, see scripts/)
cargo run --release --example cyto_final

# Optical digits (requires data download)
cargo run --release --example digits_compare
```

## Architecture

| Module | Purpose |
|---|---|
| `knn.rs` | kNN dispatch: kd-tree, HNSW, brute-force, TurboQuant |
| `kdtree.rs` | kd-tree for exact kNN (dims <= 40) |
| `hnsw.rs` | HNSW for approximate kNN (high-dim) |
| `quantize.rs` | TurboQuant 4/8-bit vector quantization |
| `fuzzy.rs` | Fuzzy simplicial set (sparse graph from kNN) |
| `sparse.rs` | CSR sparse graph |
| `spectral.rs` | Spectral, PCA, and random initialization |
| `optimize.rs` | Parallel HogWild! SGD with fast_pow |
| `model.rs` | Fit/transform model, CSV triple export |

## Dependencies

- `ndarray` — N-dimensional arrays
- `ndarray-linalg` — Eigendecomposition (OpenBLAS, statically linked)
- `ndarray-rand` — Random array generation
- `rand` — Random number generation
- `rayon` — Parallelism

No external kNN libraries. kd-tree, HNSW, and TurboQuant are implemented from scratch.
The HNSW uses heuristic neighbour selection (Malkov & Yashunin, Alg. 4) and `EF_SEARCH = 100`
with an exact 2k refine: recall@15 0.998 against brute force on 50k × 32 clustered data
(`cargo run --release --example knn_recall -- <data.csv>` measures it).

## License

TBD
