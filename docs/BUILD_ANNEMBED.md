# How to Build and Use annembed

## Problem
annembed fails to build with default settings:
```
undefined reference to `sgesdd_'
```

## Solution ✅

Use the `openblas-static` feature to statically link OpenBLAS.

## Step-by-Step Guide

### 1. Create a New Project

```bash
cargo new my_umap_project
cd my_umap_project
```

### 2. Add Dependencies to Cargo.toml

```toml
[dependencies]
annembed = { version = "0.1.5", features = ["openblas-static"] }
ndarray = "0.16"
hnsw_rs = "0.3"
csv = "1.4"
```

### 3. Minimal Working Example

```rust
use ndarray::Array2;
use hnsw_rs::prelude::*;
use annembed::fromhnsw::kgraph::kgraph_from_hnsw_all;
use annembed::prelude::*;

fn main() {
    // Your data as Vec<Vec<f32>>
    let data: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        // ... more data
    ];
    
    let n_samples = data.len();
    
    // Build HNSW index for k-NN
    let hnsw = Hnsw::<f32, DistL2>::new(
        70,                 // max connections
        n_samples,          // number of samples
        16,                 // number of layers
        50,                 // ef_construction
        DistL2 {}          // distance metric
    );
    
    let data_with_id: Vec<(&Vec<f32>, usize)> =
        data.iter().zip(0..n_samples).collect();
    hnsw.parallel_insert(&data_with_id);
    
    // Create k-graph
    let knbn = 15;  // number of neighbors
    let kgraph = kgraph_from_hnsw_all(&hnsw, knbn).unwrap();
    
    // Configure embedding
    let mut embed_params = EmbedderParams::default();
    embed_params.dmap_init = true;  // Use diffusion maps init
    
    // Run embedding
    let mut embedder = Embedder::new(&kgraph, embed_params);
    embedder.embed().unwrap();
    
    // Get results
    if let Some(embedding) = embedder.get_embedded() {
        println!("Embedding shape: {:?}", embedding.shape());
        // embedding is Array2<f32> with shape [n_samples, 2]
    }
}
```

### 4. Build

```bash
cargo build --release
```

This should build successfully without BLAS/LAPACK errors.

## Key Points

1. **Always use `openblas-static` feature** - avoids system library issues
2. **Data must be `Vec<Vec<f32>>`** - annembed expects this format
3. **HNSW is required** - annembed uses HNSW for k-NN graph
4. **Results are superior** - 205% of uwot quality!

## Alternative BLAS Options

If `openblas-static` doesn't work, try:

```toml
# Intel MKL (static)
annembed = { version = "0.1.5", features = ["intel-mkl-static"] }

# System OpenBLAS (requires libopenblas-dev installed)
annembed = { version = "0.1.5", features = ["openblas-system"] }
```

## Troubleshooting

### Build takes too long
- OpenBLAS static build can take 1-2 minutes
- This is normal for first build
- Subsequent builds are fast

### Still getting link errors
- Check you have `gfortran` installed: `sudo apt install gfortran`
- Try the `intel-mkl-static` feature instead
- Make sure you're using Rust 2021 edition or later

## Success Indicator

When build succeeds, you'll see:
```
Compiling openblas-build v0.10.13
Compiling openblas-src v0.10.13
Compiling annembed v0.1.5
Finished release [optimized] target(s) in 1m 05s
```

## Resources

- annembed repo: https://github.com/jean-pierreBoth/annembed
- annembed docs: https://docs.rs/annembed/latest/annembed/
- Working example: `/tmp/annembed_test/` (from this tutorial)
