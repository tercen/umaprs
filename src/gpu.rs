/// GPU-accelerated kNN using CUDA + cuBLAS.
///
/// ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2 * x_i · x_j
/// The dot products are computed via cuBLAS GEMM: X · X^T
/// No custom CUDA kernels needed.

use ndarray::Array2;
use rayon::prelude::*;

/// Check if CUDA is available at runtime
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cudarc::driver::CudaDevice::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(feature = "cuda")]
pub fn compute_knn_gpu(data: &Array2<f64>, k: usize) -> Array2<usize> {
    use cudarc::driver::CudaDevice;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::driver::DevicePtrMut;

    let n = data.nrows();
    let d = data.ncols();

    eprintln!("GPU kNN: {} points, {} dims", n, d);

    let dev = CudaDevice::new(0).expect("Failed to open CUDA device");
    let blas = CudaBlas::new(dev.clone()).expect("Failed to create cuBLAS handle");

    // Convert to f32
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    // Norms: ||x_i||²
    let norms: Vec<f32> = (0..n)
        .map(|i| data_f32[i * d..(i + 1) * d].iter().map(|&v| v * v).sum())
        .collect();

    // Tile size based on GPU memory (~3 GB for GTX 1650, conservative)
    // Tile needs: tile_rows * n * 4 bytes for output
    let max_tile_bytes: usize = 256 * 1024 * 1024; // 256 MB for distance tile (safe for 4GB GPUs)
    let tile_size = (max_tile_bytes / (n * 4)).max(1).min(n);

    eprintln!("  Tile size: {} (of {} total)", tile_size, n);

    // Upload full data as column-major: X^T is (d × n)
    // cuBLAS is column-major, so store X as (n × d) col-major = d rows of n elements
    let mut x_col: Vec<f32> = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            x_col[j * n + i] = data_f32[i * d + j];
        }
    }
    let d_x = dev.htod_sync_copy(&x_col).expect("Upload data");

    let mut knn_indices = Array2::zeros((n, k));

    let mut row_start = 0;
    while row_start < n {
        let row_end = (row_start + tile_size).min(n);
        let tile_rows = row_end - row_start;

        // Build tile: columns row_start..row_end of X (col-major: d × tile_rows)
        let mut tile_col: Vec<f32> = vec![0.0; tile_rows * d];
        for i in 0..tile_rows {
            for j in 0..d {
                tile_col[j * tile_rows + i] = data_f32[(row_start + i) * d + j];
            }
        }
        let d_tile = dev.htod_sync_copy(&tile_col).expect("Upload tile");

        // Output: C (n × tile_rows) col-major = dot products
        // C[j, ti] = x_j · x_{row_start + ti}
        let mut d_c = dev.alloc_zeros::<f32>(n * tile_rows).expect("Alloc output");

        // GEMM: C = X^T · X_tile
        // X^T is (n × d) col-major, X_tile is (d × tile_rows) col-major
        // C = alpha * X^T * X_tile + beta * C
        // m=n, n=tile_rows, k=d
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,  // transpose X (stored as d×n) → n×d
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: tile_rows as i32,
            k: d as i32,
            alpha: 1.0f32,
            lda: d as i32,   // leading dim of X col-major (d × n), but transposed → d
            ldb: d as i32,   // leading dim of tile (d × tile_rows)
            beta: 0.0f32,
            ldc: n as i32,   // leading dim of C (n × tile_rows)
        };

        // Hmm, the layout is tricky. Let me use a simpler approach:
        // Store X as row-major (n × d), use OP_T to treat it as (d × n)^T = (n × d)
        // Actually: cuBLAS col-major with OP_N on (n×d) stored col-major = (n×d) matrix
        // Let me just store everything in col-major properly.

        // X_cm: column-major (n × d) = d columns of n elements
        // Already stored in x_col as exactly this.
        // GEMM: C = X_cm * X_tile_cm^T? No...
        //
        // We want: C[i,j] = sum_k X[i,k] * X[row_start+j,k] = dot(row_i, row_j)
        // In col-major, X is (n × d), so:
        // C = X * X_tile^T, where X_tile is (tile_rows × d)
        // GEMM: transa=N, transb=T, m=n, n=tile_rows, k=d
        // A = X (n×d, col-major, lda=n)
        // B = X_tile (tile_rows × d, col-major, ldb=tile_rows)
        // C = (n × tile_rows, col-major, ldc=n)

        // Re-store tile as (tile_rows × d) col-major
        let mut tile_cm: Vec<f32> = vec![0.0; tile_rows * d];
        for i in 0..tile_rows {
            for j in 0..d {
                tile_cm[j * tile_rows + i] = data_f32[(row_start + i) * d + j];
            }
        }
        let d_tile2 = dev.htod_sync_copy(&tile_cm).expect("Upload tile2");

        let cfg2 = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: tile_rows as i32,
            k: d as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: tile_rows as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            <CudaBlas as Gemm<f32>>::gemm(&blas, cfg2, &d_x, &d_tile2, &mut d_c)
                .expect("cuBLAS GEMM failed");
        }

        // Download dot products
        let dots = dev.dtoh_sync_copy(&d_c).expect("Download dots");

        // Find top-k per tile row using distances
        let tile_results: Vec<Vec<usize>> = (0..tile_rows)
            .into_par_iter()
            .map(|ti| {
                let i = row_start + ti;
                let ni = norms[i];

                let mut dists: Vec<(u32, f32)> = (0..n as u32)
                    .filter(|&j| j as usize != i)
                    .map(|j| {
                        let dot = dots[j as usize + ti * n];
                        let dist_sq = (ni + norms[j as usize] - 2.0 * dot).max(0.0);
                        (j, dist_sq)
                    })
                    .collect();

                dists.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
                dists.truncate(k);
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                dists.iter().take(k).map(|&(nb, _)| nb as usize).collect()
            })
            .collect();

        for (ti, neighbors) in tile_results.iter().enumerate() {
            let i = row_start + ti;
            for (idx, &nb) in neighbors.iter().enumerate() {
                knn_indices[[i, idx]] = nb;
            }
        }

        row_start = row_end;
    }

    knn_indices
}

#[cfg(not(feature = "cuda"))]
pub fn compute_knn_gpu(_data: &Array2<f64>, _k: usize) -> Array2<usize> {
    panic!("GPU kNN requires the 'cuda' feature: cargo build --features cuda")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_available_check() {
        let _ = cuda_available();
    }
}
