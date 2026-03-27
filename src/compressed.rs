/// Compressed UMAP pipeline — runs entirely on TurboQuant-compressed vectors.
/// Original data is discarded after compression. For massive datasets.
///
/// Pipeline:
///   1. Compress data → QuantizedData (16x or 8x compression)
///   2. kNN from compressed distances (approx_dist_sq with QJL correction)
///   3. Fuzzy simplicial set from compressed distances (sigma/rho from TQ distances)
///   4. Spectral/PCA init (uses compressed data dequantized on-the-fly)
///   5. SGD optimization (operates on 2D embedding, no high-dim data needed)

use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::quantize::{QuantizedData, QuantBits};
use crate::sparse::SparseGraph;

/// Compute kNN entirely from compressed distances. No exact refinement.
pub fn knn_compressed(qdata: &QuantizedData, k: usize) -> Array2<usize> {
    let n = qdata.n_samples;
    let mut knn_indices = Array2::zeros((n, k));

    knn_indices
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, qdata.approx_dist_sq(i, j)))
                .collect();

            dists.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k);
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (idx, &(neighbor, _)) in dists.iter().take(k).enumerate() {
                row[idx] = neighbor;
            }
        });

    knn_indices
}

/// Compute fuzzy simplicial set using compressed distances for sigma/rho.
/// Same algorithm as fuzzy.rs but uses approx_dist_sq instead of exact Euclidean.
pub fn fuzzy_compressed(
    knn_indices: &Array2<usize>,
    qdata: &QuantizedData,
    k: usize,
) -> SparseGraph {
    let n = qdata.n_samples;

    // Compute rho and sigma using TQ distances
    let mut rhos = Vec::with_capacity(n);
    let mut sigmas = Vec::with_capacity(n);

    for i in 0..n {
        let neighbors = knn_indices.row(i);

        // Distances to neighbors via compressed representation
        let mut distances: Vec<f64> = neighbors
            .iter()
            .map(|&j| (qdata.approx_dist_sq(i, j) as f64).sqrt())
            .collect();
        distances.insert(0, 0.0); // distance to self

        let target = (k as f64).ln();
        let (rho, sigma) = smooth_knn_distances(&distances, target);
        rhos.push(rho);
        sigmas.push(sigma);
    }

    // Compute membership strengths using TQ distances
    let mut directed: HashMap<(usize, usize), f64> = HashMap::new();

    for i in 0..n {
        let neighbors = knn_indices.row(i);
        let rho = rhos[i];
        let sigma = sigmas[i];

        for &j in neighbors.iter() {
            let dist = (qdata.approx_dist_sq(i, j) as f64).sqrt();
            let val = (-((dist - rho).max(0.0) / sigma.max(1e-10))).exp();
            directed.insert((i, j), val);
        }
    }

    // Symmetrize: a + b - a*b
    let mut symmetric: HashMap<(usize, usize), f64> = HashMap::new();
    for (&(i, j), &a) in &directed {
        if symmetric.contains_key(&(i, j)) { continue; }
        let b = directed.get(&(j, i)).copied().unwrap_or(0.0);
        let val = a + b - a * b;
        if val > 1e-8 {
            symmetric.insert((i, j), val);
            symmetric.insert((j, i), val);
        }
    }

    let mut rows = Vec::with_capacity(symmetric.len());
    let mut cols = Vec::with_capacity(symmetric.len());
    let mut vals = Vec::with_capacity(symmetric.len());
    for (&(i, j), &v) in &symmetric {
        rows.push(i);
        cols.push(j);
        vals.push(v);
    }

    SparseGraph::from_triplets(n, &rows, &cols, &vals)
}

/// Binary search for sigma in smooth kNN distances
fn smooth_knn_distances(distances: &[f64], target: f64) -> (f64, f64) {
    let rho = distances.get(1).copied().unwrap_or(0.0).max(0.0);

    let mut lo = 0.0;
    let mut hi = 1e10;
    let mut mid = 1.0;

    for _ in 0..64 {
        mid = (lo + hi) / 2.0;
        let mut val = 0.0;
        for &d in distances.iter().skip(1) {
            val += (-((d - rho).max(0.0) / mid)).exp();
        }
        if (val - target).abs() < 1e-5 { break; }
        if val > target { hi = mid; } else { lo = mid; }
    }

    (rho, mid)
}

/// PCA initialization from dequantized compressed data (on-the-fly decompression).
/// Only decompresses one vector at a time — never materializes full f64 matrix.
pub fn pca_compressed(
    qdata: &QuantizedData,
    n_components: usize,
    random_state: Option<u64>,
) -> Array2<f64> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use ndarray_rand::rand_distr::{StandardNormal, Distribution};

    let n = qdata.n_samples;
    let d = qdata.n_dims;

    // Compute means by streaming through dequantized vectors
    let mut means = vec![0.0f64; d];
    for i in 0..n {
        let decoded = qdata.decode(i);
        for j in 0..d {
            means[j] += decoded[j] as f64;
        }
    }
    for m in means.iter_mut() { *m /= n as f64; }

    // Covariance matrix (d×d) — stream through dequantized vectors
    let mut cov = vec![0.0f64; d * d];
    for i in 0..n {
        let decoded = qdata.decode(i);
        for j in 0..d {
            let vj = decoded[j] as f64 - means[j];
            for k in j..d {
                let vk = decoded[k] as f64 - means[k];
                let val = vj * vk;
                cov[j * d + k] += val;
                if j != k { cov[k * d + j] += val; }
            }
        }
    }
    let inv_n = 1.0 / (n as f64 - 1.0);
    for v in cov.iter_mut() { *v *= inv_n; }

    // Eigendecomposition of d×d covariance
    let cov_arr = Array2::from_shape_vec((d, d), cov).unwrap();
    use ndarray_linalg::{Eigh, UPLO};

    match cov_arr.eigh(UPLO::Lower) {
        Ok((eigenvalues, eigenvectors)) => {
            let eig_vec: Vec<f64> = eigenvalues.to_vec();
            let mut indices: Vec<usize> = (0..d).collect();
            indices.sort_by(|&a, &b| eig_vec[b].partial_cmp(&eig_vec[a]).unwrap());

            // Project each dequantized vector onto top eigenvectors
            let mut embedding = Array2::zeros((n, n_components));
            for i in 0..n {
                let decoded = qdata.decode(i);
                for comp in 0..n_components {
                    let eig_idx = indices[comp];
                    let mut val = 0.0f64;
                    for j in 0..d {
                        val += (decoded[j] as f64 - means[j]) * eigenvectors[[j, eig_idx]];
                    }
                    embedding[[i, comp]] = val;
                }
            }

            // Scale to std dev = 1
            for comp in 0..n_components {
                let col_mean = embedding.column(comp).sum() / n as f64;
                let var: f64 = embedding.column(comp).iter()
                    .map(|&x| (x - col_mean).powi(2))
                    .sum::<f64>() / (n as f64 - 1.0);
                let std = var.sqrt().max(1e-10);
                for i in 0..n {
                    embedding[[i, comp]] /= std;
                }
            }

            // Add noise
            let mut rng = match random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_entropy(),
            };
            let dist = StandardNormal;
            for i in 0..n {
                for j in 0..n_components {
                    let noise: f64 = dist.sample(&mut rng);
                    embedding[[i, j]] += noise * 0.0001;
                }
            }

            embedding
        }
        Err(_) => {
            // Fallback to random
            crate::spectral::random_initialization(n, n_components, random_state)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_compressed_pipeline() {
        // Small test: 20 points, 16 dims, two clusters
        let mut data_vec = vec![0.0f64; 20 * 16];
        for i in 0..10 {
            for j in 0..8 { data_vec[i * 16 + j] = 10.0 + i as f64 * 0.5; }
        }
        for i in 10..20 {
            for j in 8..16 { data_vec[i * 16 + j] = 10.0 + i as f64 * 0.5; }
        }
        let data = Array2::from_shape_vec((20, 16), data_vec).unwrap();

        // Compress
        let qdata = QuantizedData::encode_with_bits(&data, 42, QuantBits::Eight);

        // kNN from compressed
        let knn = knn_compressed(&qdata, 3);
        assert_eq!(knn.shape(), &[20, 3]);

        // Point 0's neighbors should be in cluster A
        assert!(knn[[0, 0]] < 10, "got {}", knn[[0, 0]]);

        // Fuzzy from compressed
        let graph = fuzzy_compressed(&knn, &qdata, 3);
        assert!(graph.nnz() > 0);
    }
}
