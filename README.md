# umaprs

A fast Rust implementation of UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.

2-3x faster than R uwot with comparable or better quality on real datasets. Minimal dependencies, self-contained binary (~1.5 MB).

## Performance

Tested on the Levine CyTOF dataset (50,000 cells, 32 protein markers, 14 populations):

| | umaprs | R uwot |
|---|---|---|
| Time | **6.5s** | 15.2s |
| Separation | **8.85** | 8.48 |
| Quality | **104%** | 100% |

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
    .random_state(42)            // reproducibility seed
```

## kNN Methods

| Method | Type | Best for |
|---|---|---|
| `KdTree` | Exact | dims <= 40 (default for this range) |
| `Hnsw` | Approximate | dims > 40 |
| `BruteForce` | Exact | n <= 500 (default for this range) |
| `TurboQuant4KdTree` | Approximate | Memory-constrained, high-dim (experimental) |
| `TurboQuant8KdTree` | Approximate | Memory-constrained, moderate-dim (experimental) |

See [docs/turboquant.md](docs/turboquant.md) for details on TurboQuant methods.

## Fit / Transform

For large datasets, fit on a subset and transform the rest:

```rust
// Automatic: fit on 10%, transform remaining 90%
let embedding = UMAP::new()
    .train_size(0.1)
    .fit_transform(&data);

// Manual: get model for later use
let (embedding, model) = UMAP::new()
    .model_format(ModelFormat::Csv("model.csv".into()))
    .fit(&data);

// Transform new data with saved model
let model = UmapModel::load_triples_csv("model.csv").unwrap();
let new_embedding = model.transform(&new_data);
```

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

## License

TBD
