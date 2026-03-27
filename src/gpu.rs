/// GPU-accelerated kNN via CUDA.
/// Kernels written in CUDA C, compiled at runtime via NVRTC.
/// No nvcc needed at build time.

use ndarray::Array2;
use rayon::prelude::*;

pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    { cudarc::driver::CudaDevice::new(0).is_ok() }
    #[cfg(not(feature = "cuda"))]
    { false }
}

#[cfg(feature = "cuda")]
mod inner {
    use super::*;
    use cudarc::driver::{CudaDevice, CudaFunction, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::compile_ptx;
    use std::sync::Arc;

    const CUDA_SRC: &str = include_str!("kernels.cu");

    struct GpuKernels {
        dev: Arc<CudaDevice>,
        tq4_dot: CudaFunction,
        tq8_dot: CudaFunction,
        topk: CudaFunction,
        f32_dot: CudaFunction,
    }

    fn load_kernels() -> GpuKernels {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let ptx = compile_ptx(CUDA_SRC).expect("NVRTC compilation failed");
        dev.load_ptx(ptx, "umaprs", &["tq4_dot", "tq8_dot", "topk", "f32_dot"])
            .expect("Load kernels");
        GpuKernels {
            tq4_dot: dev.get_func("umaprs", "tq4_dot").unwrap(),
            tq8_dot: dev.get_func("umaprs", "tq8_dot").unwrap(),
            topk: dev.get_func("umaprs", "topk").unwrap(),
            f32_dot: dev.get_func("umaprs", "f32_dot").unwrap(),
            dev,
        }
    }

    /// GPU brute-force kNN on f32 data using CUDA dot product + topk kernels.
    /// No cuBLAS dependency.
    pub fn compute_knn_gpu(data: &Array2<f64>, k: usize) -> Array2<usize> {
        let n = data.nrows();
        let d = data.ncols();
        eprintln!("GPU kNN: {} points, {} dims", n, d);

        let kern = load_kernels();
        let dev = &kern.dev;

        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let norms: Vec<f32> = (0..n)
            .map(|i| data_f32[i * d..(i + 1) * d].iter().map(|v| v * v).sum())
            .collect();

        // Upload all data (row-major f32)
        let d_data = dev.htod_sync_copy(&data_f32).expect("Upload data");
        let d_norms = dev.htod_sync_copy(&norms).expect("Upload norms");

        let tile_bytes: usize = 512 * 1024 * 1024;
        let tile_size = (tile_bytes / (n * 4)).max(1).min(n);
        eprintln!("  Tile: {} rows, {} tiles", tile_size, (n + tile_size - 1) / tile_size);

        let mut knn_indices = Array2::zeros((n, k));
        let mut row_start = 0;

        while row_start < n {
            let row_end = (row_start + tile_size).min(n);
            let tile_rows = row_end - row_start;

            // Upload tile
            let tile_data = &data_f32[row_start * d..row_end * d];
            let d_tile = dev.htod_sync_copy(tile_data).expect("Upload tile");
            let mut d_dots = dev.alloc_zeros::<f32>(tile_rows * n).expect("Alloc dots");

            // Kernel 1: dot products
            let dot_cfg = LaunchConfig {
                grid_dim: (((n as u32) + 15) / 16, ((tile_rows as u32) + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let (ptr_t, ptr_d, mut ptr_o) = (*d_tile.device_ptr(), *d_data.device_ptr(), *d_dots.device_ptr_mut());
            let (arg_tr, arg_n, arg_d) = (tile_rows as i32, n as i32, d as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &ptr_t as *const _ as *mut _, &ptr_d as *const _ as *mut _,
                &mut ptr_o as *mut _ as *mut _,
                &arg_tr as *const _ as *mut _, &arg_n as *const _ as *mut _, &arg_d as *const _ as *mut _,
            ];
            unsafe { kern.f32_dot.clone().launch(dot_cfg, &mut args).expect("f32_dot failed"); }

            // Kernel 2: topk on GPU
            let tile_norms = &norms[row_start..row_end];
            let d_tile_norms = dev.htod_sync_copy(tile_norms).expect("Upload tile norms");
            let mut d_topk = dev.alloc_zeros::<u32>(tile_rows * k).expect("Alloc topk");

            let topk_cfg = LaunchConfig {
                grid_dim: (tile_rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let ptr_dots = *d_dots.device_ptr();
            let ptr_tn = *d_tile_norms.device_ptr();
            let ptr_an = *d_norms.device_ptr();
            let mut ptr_tk = *d_topk.device_ptr_mut();
            let (arg_k, arg_off) = (k as i32, row_start as i32);
            let inv_d = 0.0f32; // unused in f32 mode
            let mode = 0i32;    // f32 mode: dist = ni + nj - 2*dot

            let mut topk_args: Vec<*mut std::ffi::c_void> = vec![
                &ptr_dots as *const _ as *mut _, &ptr_tn as *const _ as *mut _,
                &ptr_an as *const _ as *mut _, &mut ptr_tk as *mut _ as *mut _,
                &arg_tr as *const _ as *mut _, &arg_n as *const _ as *mut _,
                &arg_k as *const _ as *mut _, &arg_off as *const _ as *mut _,
                &inv_d as *const _ as *mut _, &mode as *const _ as *mut _,
            ];
            unsafe { kern.topk.clone().launch(topk_cfg, &mut topk_args).expect("topk failed"); }

            // Download only k indices per row (tiny)
            let topk_idx: Vec<u32> = dev.dtoh_sync_copy(&d_topk).expect("Download topk");

            for ti in 0..tile_rows {
                let i = row_start + ti;
                for idx in 0..k {
                    knn_indices[[i, idx]] = topk_idx[ti * k + idx] as usize;
                }
            }

            row_start = row_end;
        }

        knn_indices
    }

    /// GPU TQ4: packed 4-bit dot products + topk, all on GPU
    pub fn compute_knn_gpu_tq4(data: &Array2<f64>, k: usize) -> Array2<usize> {
        use crate::quantize::{QuantizedData, QuantBits};

        let n = data.nrows();
        let d = data.ncols();
        eprintln!("GPU TQ4 kNN: {} points, {} dims", n, d);

        let qdata = QuantizedData::encode_with_bits(data, 42, QuantBits::Four);
        let packed = qdata.packed_data();
        let norms = qdata.norms();
        let codebook = qdata.sorted_centroids();
        let padded_dims = qdata.padded_dims();
        let bpp = padded_dims / 2;

        eprintln!("  Memory: {} KB -> {} KB ({:.1}x)",
                  n * d * 8 / 1024, qdata.memory_bytes() / 1024,
                  (n * d * 8) as f64 / qdata.memory_bytes() as f64);
        eprintln!("  GPU data: {} KB (vs {} KB f32)", packed.len() / 1024, n * d * 4 / 1024);

        let kern = load_kernels();
        let dev = &kern.dev;

        let d_all_packed = dev.htod_sync_copy(packed).expect("Upload packed");
        let d_codebook = dev.htod_sync_copy(&codebook).expect("Upload codebook");
        let d_norms = dev.htod_sync_copy(norms).expect("Upload norms");

        let tile_bytes: usize = 512 * 1024 * 1024;
        let tile_size = (tile_bytes / (n * 4)).max(1).min(n);
        eprintln!("  Tile: {} rows, {} tiles", tile_size, (n + tile_size - 1) / tile_size);

        let inv_d = 1.0f32 / padded_dims as f32;
        let refine_k = k * 2; // get 2k from GPU, refine on CPU
        let mut knn_indices = Array2::zeros((n, k));
        let mut row_start = 0;

        while row_start < n {
            let row_end = (row_start + tile_size).min(n);
            let tile_rows = row_end - row_start;

            let tile_packed = &packed[row_start * bpp..row_end * bpp];
            let d_tile = dev.htod_sync_copy(tile_packed).expect("Upload tile");
            let mut d_dots = dev.alloc_zeros::<f32>(tile_rows * n).expect("Alloc dots");

            // Kernel 1: TQ4 dot products
            let dot_cfg = LaunchConfig {
                grid_dim: (((n as u32) + 15) / 16, ((tile_rows as u32) + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let ptr_a = *d_tile.device_ptr();
            let ptr_b = *d_all_packed.device_ptr();
            let mut ptr_c = *d_dots.device_ptr_mut();
            let ptr_cb = *d_codebook.device_ptr();
            let (arg_tr, arg_n, arg_dh) = (tile_rows as i32, n as i32, bpp as i32);
            let mut dot_args: Vec<*mut std::ffi::c_void> = vec![
                &ptr_a as *const _ as *mut _, &ptr_b as *const _ as *mut _,
                &mut ptr_c as *mut _ as *mut _, &ptr_cb as *const _ as *mut _,
                &arg_tr as *const _ as *mut _, &arg_n as *const _ as *mut _, &arg_dh as *const _ as *mut _,
            ];
            unsafe { kern.tq4_dot.clone().launch(dot_cfg, &mut dot_args).expect("tq4_dot failed"); }

            // Kernel 2: topk on GPU
            let tile_norms: Vec<f32> = norms[row_start..row_end].to_vec();
            let d_tile_norms = dev.htod_sync_copy(&tile_norms).expect("Upload tile norms");
            let mut d_topk = dev.alloc_zeros::<u32>(tile_rows * refine_k).expect("Alloc topk");

            let topk_cfg = LaunchConfig {
                grid_dim: (tile_rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let ptr_dots = *d_dots.device_ptr();
            let ptr_tn = *d_tile_norms.device_ptr();
            let ptr_an = *d_norms.device_ptr();
            let mut ptr_tk = *d_topk.device_ptr_mut();
            let (arg_rk, arg_off) = (refine_k as i32, row_start as i32);
            let mode = 1i32; // TQ mode
            let mut topk_args: Vec<*mut std::ffi::c_void> = vec![
                &ptr_dots as *const _ as *mut _, &ptr_tn as *const _ as *mut _,
                &ptr_an as *const _ as *mut _, &mut ptr_tk as *mut _ as *mut _,
                &arg_tr as *const _ as *mut _, &arg_n as *const _ as *mut _,
                &arg_rk as *const _ as *mut _, &arg_off as *const _ as *mut _,
                &inv_d as *const _ as *mut _, &mode as *const _ as *mut _,
            ];
            unsafe { kern.topk.clone().launch(topk_cfg, &mut topk_args).expect("topk failed"); }

            // Download tiny topk indices, refine on CPU
            let topk_idx: Vec<u32> = dev.dtoh_sync_copy(&d_topk).expect("Download topk");

            let tile_results: Vec<Vec<usize>> = (0..tile_rows)
                .into_par_iter()
                .map(|ti| {
                    let i = row_start + ti;
                    let candidates: Vec<usize> = (0..refine_k)
                        .map(|r| topk_idx[ti * refine_k + r] as usize)
                        .filter(|&j| j != i && j < n)
                        .collect();
                    let point = data.row(i);
                    let mut exact: Vec<(usize, f64)> = candidates.iter()
                        .map(|&j| {
                            let d: f64 = point.iter().zip(data.row(j).iter())
                                .map(|(&a, &b)| (a - b).powi(2)).sum();
                            (j, d)
                        })
                        .collect();
                    exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    exact.iter().take(k).map(|&(nb, _)| nb).collect()
                })
                .collect();

            for (ti, neighbors) in tile_results.iter().enumerate() {
                for (idx, &nb) in neighbors.iter().enumerate() {
                    knn_indices[[row_start + ti, idx]] = nb;
                }
            }

            row_start = row_end;
        }

        knn_indices
    }
    /// GPU TQ8: packed 8-bit dot products + topk, all on GPU
    pub fn compute_knn_gpu_tq8(data: &Array2<f64>, k: usize) -> Array2<usize> {
        use crate::quantize::{QuantizedData, QuantBits};

        let n = data.nrows();
        let d = data.ncols();
        eprintln!("GPU TQ8 kNN: {} points, {} dims", n, d);

        let qdata = QuantizedData::encode_with_bits(data, 42, QuantBits::Eight);
        let packed = qdata.packed_data();
        let norms = qdata.norms();
        let padded_dims = qdata.padded_dims();
        let bpp = padded_dims; // 1 byte per coordinate for 8-bit

        eprintln!("  Memory: {} KB -> {} KB ({:.1}x)",
                  n * d * 8 / 1024, qdata.memory_bytes() / 1024,
                  (n * d * 8) as f64 / qdata.memory_bytes() as f64);

        let kern = load_kernels();
        let dev = &kern.dev;

        let d_all_packed = dev.htod_sync_copy(packed).expect("Upload packed");
        let codebook = qdata.sorted_centroids(); // 128 centroids + mse at [128]
        let d_codebook = dev.htod_sync_copy(&codebook).expect("Upload codebook");
        let d_norms = dev.htod_sync_copy(norms).expect("Upload norms");

        let tile_bytes: usize = 512 * 1024 * 1024;
        let tile_size = (tile_bytes / (n * 4)).max(1).min(n);
        eprintln!("  Tile: {} rows, {} tiles", tile_size, (n + tile_size - 1) / tile_size);

        let inv_d = 1.0f32 / padded_dims as f32;
        let refine_k = k * 2;
        let mut knn_indices = Array2::zeros((n, k));
        let mut row_start = 0;

        while row_start < n {
            let row_end = (row_start + tile_size).min(n);
            let tile_rows = row_end - row_start;

            let tile_packed = &packed[row_start * bpp..row_end * bpp];
            let d_tile = dev.htod_sync_copy(tile_packed).expect("Upload tile");
            let mut d_dots = dev.alloc_zeros::<f32>(tile_rows * n).expect("Alloc dots");

            // TQ8 dot kernel (codebook lookup, same pattern as TQ4)
            let dot_cfg = LaunchConfig {
                grid_dim: (((n as u32) + 15) / 16, ((tile_rows as u32) + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let ptr_a = *d_tile.device_ptr();
            let ptr_b = *d_all_packed.device_ptr();
            let mut ptr_c = *d_dots.device_ptr_mut();
            let ptr_cb = *d_codebook.device_ptr();
            let (arg_tr, arg_n, arg_d) = (tile_rows as i32, n as i32, padded_dims as i32);
            let mut dot_args: Vec<*mut std::ffi::c_void> = vec![
                &ptr_a as *const _ as *mut _, &ptr_b as *const _ as *mut _,
                &mut ptr_c as *mut _ as *mut _, &ptr_cb as *const _ as *mut _,
                &arg_tr as *const _ as *mut _, &arg_n as *const _ as *mut _,
                &arg_d as *const _ as *mut _,
            ];
            unsafe { kern.tq8_dot.clone().launch(dot_cfg, &mut dot_args).expect("tq8_dot failed"); }

            // topk on GPU
            let tile_norms: Vec<f32> = norms[row_start..row_end].to_vec();
            let d_tile_norms = dev.htod_sync_copy(&tile_norms).expect("Upload tile norms");
            let mut d_topk = dev.alloc_zeros::<u32>(tile_rows * refine_k).expect("Alloc topk");

            let topk_cfg = LaunchConfig {
                grid_dim: (tile_rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let ptr_dots = *d_dots.device_ptr();
            let ptr_tn = *d_tile_norms.device_ptr();
            let ptr_an = *d_norms.device_ptr();
            let mut ptr_tk = *d_topk.device_ptr_mut();
            let (arg_rk, arg_off) = (refine_k as i32, row_start as i32);
            let mode = 1i32; // TQ mode
            let mut topk_args: Vec<*mut std::ffi::c_void> = vec![
                &ptr_dots as *const _ as *mut _, &ptr_tn as *const _ as *mut _,
                &ptr_an as *const _ as *mut _, &mut ptr_tk as *mut _ as *mut _,
                &arg_tr as *const _ as *mut _, &arg_n as *const _ as *mut _,
                &arg_rk as *const _ as *mut _, &arg_off as *const _ as *mut _,
                &inv_d as *const _ as *mut _, &mode as *const _ as *mut _,
            ];
            unsafe { kern.topk.clone().launch(topk_cfg, &mut topk_args).expect("topk failed"); }

            // Download tiny topk, refine on CPU
            let topk_idx: Vec<u32> = dev.dtoh_sync_copy(&d_topk).expect("Download topk");

            let tile_results: Vec<Vec<usize>> = (0..tile_rows)
                .into_par_iter()
                .map(|ti| {
                    let i = row_start + ti;
                    let candidates: Vec<usize> = (0..refine_k)
                        .map(|r| topk_idx[ti * refine_k + r] as usize)
                        .filter(|&j| j != i && j < n)
                        .collect();
                    let point = data.row(i);
                    let mut exact: Vec<(usize, f64)> = candidates.iter()
                        .map(|&j| {
                            let d: f64 = point.iter().zip(data.row(j).iter())
                                .map(|(&a, &b)| (a - b).powi(2)).sum();
                            (j, d)
                        })
                        .collect();
                    exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    exact.iter().take(k).map(|&(nb, _)| nb).collect()
                })
                .collect();

            for (ti, neighbors) in tile_results.iter().enumerate() {
                for (idx, &nb) in neighbors.iter().enumerate() {
                    knn_indices[[row_start + ti, idx]] = nb;
                }
            }

            row_start = row_end;
        }

        knn_indices
    }
}

#[cfg(feature = "cuda")]
pub use inner::{compute_knn_gpu, compute_knn_gpu_tq4, compute_knn_gpu_tq8};

#[cfg(not(feature = "cuda"))]
pub fn compute_knn_gpu(_data: &Array2<f64>, _k: usize) -> Array2<usize> {
    panic!("GPU kNN requires the 'cuda' feature: cargo build --features cuda")
}
#[cfg(not(feature = "cuda"))]
pub fn compute_knn_gpu_tq4(_data: &Array2<f64>, _k: usize) -> Array2<usize> {
    panic!("GPU TQ4 requires the 'cuda' feature: cargo build --features cuda")
}
#[cfg(not(feature = "cuda"))]
pub fn compute_knn_gpu_tq8(_data: &Array2<f64>, _k: usize) -> Array2<usize> {
    panic!("GPU TQ8 requires the 'cuda' feature: cargo build --features cuda")
}
#[cfg(not(feature = "cuda"))]
pub fn compute_knn_gpu_tile(_data: &Array2<f64>, _k: usize, _tile_mb: usize) -> Array2<usize> {
    panic!("GPU kNN requires the 'cuda' feature")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cuda_check() { let _ = cuda_available(); }
}
