use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

use crate::hnsw::Hnsw;
use crate::kdtree::KdTree;
use crate::quantize::{QuantizedData, QuantBits};

/// Threshold for switching from brute-force to tree-based methods
const TREE_THRESHOLD: usize = 500;

/// Max dimensions for kd-tree (above this, HNSW is better)
const KDTREE_MAX_DIMS: usize = 40;

/// Minimum dimensions for TurboQuant to be effective
const QUANT_MIN_DIMS: usize = 16;

/// Compute k-nearest neighbors for each point.
/// Strategy:
///   - Small datasets (<=500): exact brute-force
///   - Large + low-dim (<=40): kd-tree (exact, like uwot's FNN)
///   - Large + high-dim (>40): HNSW (approximate)
pub fn compute_knn_graph(data: &Array2<f64>, k: usize) -> Array2<usize> {
    let n_samples = data.nrows();
    let n_dims = data.ncols();

    if n_samples <= TREE_THRESHOLD {
        compute_knn_bruteforce(data, k)
    } else if n_dims <= KDTREE_MAX_DIMS {
        eprintln!(
            "Using kd-tree exact nearest neighbors ({} points, {} dims)",
            n_samples, n_dims
        );
        compute_knn_kdtree(data, k)
    } else {
        eprintln!(
            "Using HNSW approximate nearest neighbors ({} points, {} dims)",
            n_samples, n_dims
        );
        compute_knn_hnsw_f32(data, k)
    }
}

/// Brute-force kNN — exact, O(n²), good for small datasets
pub fn compute_knn_bruteforce(data: &Array2<f64>, k: usize) -> Array2<usize> {
    let n_samples = data.nrows();
    let mut knn_indices = Array2::zeros((n_samples, k));

    knn_indices
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            let point = data.row(i);
            let mut distances: Vec<(usize, f64)> = (0..n_samples)
                .filter(|&j| i != j)
                .map(|j| (j, euclidean_distance(point, data.row(j))))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for (idx, &(neighbor, _)) in distances.iter().take(k).enumerate() {
                row[idx] = neighbor;
            }
        });

    knn_indices
}

/// TurboQuant + HNSW (4-bit)
pub fn compute_knn_quant_hnsw(data: &Array2<f64>, k: usize) -> Array2<usize> {
    compute_knn_quant_hnsw_bits(data, k, QuantBits::Four)
}

/// TurboQuant + HNSW (8-bit)
pub fn compute_knn_quant_hnsw_8bit(data: &Array2<f64>, k: usize) -> Array2<usize> {
    compute_knn_quant_hnsw_bits(data, k, QuantBits::Eight)
}

fn compute_knn_quant_hnsw_bits(data: &Array2<f64>, k: usize, bits: QuantBits) -> Array2<usize> {
    let n_samples = data.nrows();

    let qdata = QuantizedData::encode_with_bits(data, 42, bits);

    let original_bytes = n_samples * data.ncols() * 8;
    let quant_bytes = qdata.memory_bytes();
    eprintln!(
        "  Memory: {} KB -> {} KB ({:.1}x compression)",
        original_bytes / 1024,
        quant_bytes / 1024,
        original_bytes as f64 / quant_bytes as f64
    );

    // Build HNSW with squared distance — skip sqrt, ordering is preserved
    let dist_sq_fn = |i: u32, j: u32| -> f32 {
        qdata.approx_dist_sq(i as usize, j as usize)
    };
    let hnsw = Hnsw::build(n_samples, &dist_sq_fn, 42);

    // Search: approximate top-2k from HNSW, refine with exact distances
    let refine_k = (k * 2).min(n_samples - 1);
    let mut knn_indices = Array2::zeros((n_samples, k));

    knn_indices
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            let results = hnsw.search(i as u32, refine_k + 1, &dist_sq_fn);

            let candidates: Vec<usize> = results
                .iter()
                .map(|&(id, _)| id as usize)
                .filter(|&j| j != i)
                .collect();

            let point = data.row(i);
            let mut exact_dists: Vec<(usize, f64)> = candidates
                .iter()
                .map(|&j| (j, euclidean_distance(point, data.row(j))))
                .collect();
            exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (idx, &(neighbor, _)) in exact_dists.iter().take(k).enumerate() {
                row[idx] = neighbor;
            }
        });

    knn_indices
}

/// kd-tree exact kNN — O(n log n) build, O(k log n) query. Exact results.
pub fn compute_knn_kdtree(data: &Array2<f64>, k: usize) -> Array2<usize> {
    let n_samples = data.nrows();
    let n_dims = data.ncols();

    let flat: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let tree = KdTree::build(&flat, n_samples, n_dims);

    let mut knn_indices = Array2::zeros((n_samples, k));
    knn_indices
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            let results = tree.knn(i, k);
            for (idx, &(nb, _)) in results.iter().enumerate() {
                row[idx] = nb as usize;
            }
        });

    knn_indices
}

/// TurboQuant + kd-tree: quantize, dequantize to f32, exact kNN on quantized repr.
pub fn compute_knn_quant_kdtree(data: &Array2<f64>, k: usize, bits: QuantBits) -> Array2<usize> {
    let n_samples = data.nrows();
    let qdata = QuantizedData::encode_with_bits(data, 42, bits);

    let original_bytes = n_samples * data.ncols() * 8;
    let quant_bytes = qdata.memory_bytes();
    eprintln!(
        "  TQ {:?} + kd-tree: {} KB -> {} KB ({:.1}x compression)",
        bits, original_bytes / 1024, quant_bytes / 1024,
        original_bytes as f64 / quant_bytes as f64
    );

    // Dequantize all vectors to f32 for kd-tree
    let n_dims = data.ncols();
    let mut flat = vec![0.0f32; n_samples * n_dims];
    for i in 0..n_samples {
        let decoded = qdata.decode(i);
        for j in 0..n_dims {
            flat[i * n_dims + j] = decoded[j];
        }
    }

    let tree = KdTree::build(&flat, n_samples, n_dims);

    // kd-tree gives exact kNN on the dequantized data,
    // then refine with exact f64 distances on original data
    let refine_k = (k * 2).min(n_samples - 1);
    let mut knn_indices = Array2::zeros((n_samples, k));
    knn_indices
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            let candidates = tree.knn(i, refine_k);

            let point = data.row(i);
            let mut exact_dists: Vec<(usize, f64)> = candidates.iter()
                .map(|&(nb, _)| (nb as usize, euclidean_distance(point, data.row(nb as usize))))
                .collect();
            exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (idx, &(neighbor, _)) in exact_dists.iter().take(k).enumerate() {
                row[idx] = neighbor;
            }
        });

    knn_indices
}

/// TQ 4-bit + kd-tree
pub fn compute_knn_quant4_kdtree(data: &Array2<f64>, k: usize) -> Array2<usize> {
    compute_knn_quant_kdtree(data, k, QuantBits::Four)
}

/// TQ 8-bit + kd-tree
pub fn compute_knn_quant8_kdtree(data: &Array2<f64>, k: usize) -> Array2<usize> {
    compute_knn_quant_kdtree(data, k, QuantBits::Eight)
}

/// Plain HNSW with f32 distances (no quantization)
pub fn compute_knn_hnsw_f32(data: &Array2<f64>, k: usize) -> Array2<usize> {
    let n_samples = data.nrows();
    let n_dims = data.ncols();

    // Convert to flat f32 for fast distance
    let flat: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let dist_fn = move |i: u32, j: u32| -> f32 {
        let a = i as usize * n_dims;
        let b = j as usize * n_dims;
        let mut sum = 0.0f32;
        for d in 0..n_dims {
            let diff = unsafe { flat.get_unchecked(a + d) - flat.get_unchecked(b + d) };
            sum += diff * diff;
        }
        sum // squared distance — sqrt not needed for ordering
    };

    let hnsw = Hnsw::build(n_samples, &dist_fn, 42);

    // Get 2k candidates from HNSW, refine with exact f64 distances
    let refine_k = (k * 2).min(n_samples - 1);
    let mut knn_indices = Array2::zeros((n_samples, k));
    knn_indices
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            let results = hnsw.search(i as u32, refine_k + 1, &dist_fn);

            let candidates: Vec<usize> = results.iter()
                .map(|&(nb, _)| nb as usize)
                .filter(|&j| j != i)
                .collect();

            let point = data.row(i);
            let mut exact_dists: Vec<(usize, f64)> = candidates.iter()
                .map(|&j| (j, euclidean_distance(point, data.row(j))))
                .collect();
            exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (idx, &(neighbor, _)) in exact_dists.iter().take(k).enumerate() {
                row[idx] = neighbor;
            }
        });

    knn_indices
}

fn euclidean_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_euclidean_distance() {
        let a = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let b = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
        let dist = euclidean_distance(a.row(0), b.row(0));
        assert!((dist - 1.732).abs() < 0.01);
    }

    #[test]
    fn test_knn_bruteforce() {
        let data = Array2::from_shape_vec((5, 2), vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0, 11.0, 10.0,
        ]).unwrap();
        let knn = compute_knn_bruteforce(&data, 2);
        assert_eq!(knn.shape(), &[5, 2]);
        assert!(knn[[0, 0]] == 1 || knn[[0, 0]] == 2);
        assert_eq!(knn[[3, 0]], 4);
    }

    #[test]
    fn test_knn_quant_hnsw() {
        let mut data_vec = vec![0.0f64; 10 * 16];
        for i in 0..5 { data_vec[i * 16] = i as f64 * 0.1; data_vec[i * 16 + 1] = i as f64 * 0.1; }
        for i in 5..10 { data_vec[i * 16] = 100.0 + i as f64 * 0.1; data_vec[i * 16 + 1] = 100.0 + i as f64 * 0.1; }
        let data = Array2::from_shape_vec((10, 16), data_vec).unwrap();
        let knn = compute_knn_quant_hnsw(&data, 2);
        assert_eq!(knn.shape(), &[10, 2]);
        assert!(knn[[0, 0]] < 5, "got {}", knn[[0, 0]]);
        assert!(knn[[5, 0]] >= 5, "got {}", knn[[5, 0]]);
    }
}
