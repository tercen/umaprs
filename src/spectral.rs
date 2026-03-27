use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, StandardNormal, Distribution};
use rand::SeedableRng;
use rand::rngs::StdRng;
use ndarray_linalg::{Eigh, UPLO};

use crate::sparse::SparseGraph;

/// Above this threshold, skip spectral init and use random initialization
/// (eigendecomposition requires a dense n×n matrix)
const SPECTRAL_DENSE_THRESHOLD: usize = 2000;

/// Initialize embedding using spectral layout with Laplacian eigenvectors.
/// Falls back to PCA for large datasets (like uwot's "spca" fallback).
pub fn spectral_layout(
    graph: &SparseGraph,
    n_components: usize,
    random_state: Option<u64>,
) -> Array2<f64> {
    spectral_layout_with_data(graph, n_components, random_state, None)
}

/// Spectral layout with optional data for PCA fallback
pub fn spectral_layout_with_data(
    graph: &SparseGraph,
    n_components: usize,
    random_state: Option<u64>,
    data: Option<&Array2<f64>>,
) -> Array2<f64> {
    let n_samples = graph.n_nodes;

    if n_samples > SPECTRAL_DENSE_THRESHOLD {
        // Use PCA initialization if data is available (like uwot's "spca" fallback)
        if let Some(data) = data {
            eprintln!("Using PCA initialization ({} samples, {} dims)",
                      n_samples, data.ncols());
            return pca_initialization(data, n_components, random_state);
        }
        eprintln!(
            "Dataset too large for spectral init ({} > {}), using random initialization",
            n_samples, SPECTRAL_DENSE_THRESHOLD
        );
        return random_initialization(n_samples, n_components, random_state);
    }

    // Convert to dense for eigendecomposition
    let dense = graph.to_dense();
    let laplacian = compute_normalized_laplacian(&dense);

    match compute_spectral_embedding(&laplacian, n_components, random_state) {
        Ok(emb) => emb,
        Err(e) => {
            eprintln!("Warning: Spectral initialization failed ({}), using random initialization", e);
            random_initialization(n_samples, n_components, random_state)
        }
    }
}

/// Compute normalized graph Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
fn compute_normalized_laplacian(graph: &Array2<f64>) -> Array2<f64> {
    let n = graph.nrows();
    let mut laplacian = Array2::zeros((n, n));

    // Compute degree for each node
    let mut degrees = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            degrees[i] += graph[[i, j]];
        }
        // Avoid division by zero
        if degrees[i] < 1e-10 {
            degrees[i] = 1.0;
        }
    }

    // Compute D^(-1/2)
    let d_inv_sqrt: Vec<f64> = degrees.iter().map(|&d| 1.0 / d.sqrt()).collect();

    // L = I - D^(-1/2) * A * D^(-1/2)
    for i in 0..n {
        for j in 0..n {
            if i == j {
                laplacian[[i, j]] = 1.0 - d_inv_sqrt[i] * graph[[i, j]] * d_inv_sqrt[j];
            } else {
                laplacian[[i, j]] = -d_inv_sqrt[i] * graph[[i, j]] * d_inv_sqrt[j];
            }
        }
    }

    laplacian
}

/// Compute spectral embedding from Laplacian eigenvectors
/// Uses proper eigendecomposition to find smallest eigenvectors
fn compute_spectral_embedding(
    laplacian: &Array2<f64>,
    n_components: usize,
    random_state: Option<u64>,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let n_samples = laplacian.nrows();

    // Compute all eigenvalues and eigenvectors using symmetric eigendecomposition
    let (eigenvalues, eigenvectors) = laplacian.clone().eigh(UPLO::Lower)?;

    // Find indices of smallest eigenvalues
    // Sort eigenvalue indices by value (ascending)
    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Debug: print first few eigenvalues
    eprintln!("Spectral init: First 5 eigenvalues:");
    for i in 0..5.min(eigenvalues.len()) {
        eprintln!("  λ[{}] = {:.6}", i, eigenvalues[indices[i]]);
    }

    // Skip the first (smallest) eigenvalue which should be ~0 (constant eigenvector)
    // Take the next n_components eigenvectors (corresponding to smallest non-zero eigenvalues)
    let start_idx = 1.min(indices.len() - 1);
    let end_idx = (start_idx + n_components).min(indices.len());

    eprintln!("Using eigenvectors {} to {} (eigenvalues {:.6} to {:.6})",
              start_idx, end_idx-1, eigenvalues[indices[start_idx]], eigenvalues[indices[end_idx-1]]);

    // Build embedding from eigenvectors
    let mut embedding = Array2::zeros((n_samples, n_components));

    for (comp_idx, &eig_idx) in indices[start_idx..end_idx].iter().enumerate() {
        if comp_idx >= n_components {
            break;
        }

        // Extract eigenvector column
        for i in 0..n_samples {
            embedding[[i, comp_idx]] = eigenvectors[[i, eig_idx]];
        }
    }

    // Add small noise for stability (like uwot does)
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let noise_scale = 0.0001;
    let dist = StandardNormal;
    for i in 0..n_samples {
        for j in 0..n_components {
            let noise: f64 = dist.sample(&mut rng);
            embedding[[i, j]] += noise * noise_scale;
        }
    }

    // Scale to reasonable range (like uwot: typically [-10, 10])
    let max_abs = embedding.iter()
        .map(|&x| x.abs())
        .fold(0.0f64, f64::max)
        .max(1e-10);

    let scale = 10.0 / max_abs;
    embedding.mapv_inplace(|x| x * scale);

    Ok(embedding)
}

/// Fallback: random initialization
pub(crate) fn random_initialization(
    n_samples: usize,
    n_components: usize,
    random_state: Option<u64>,
) -> Array2<f64> {
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut embedding = Array2::random_using(
        (n_samples, n_components),
        Uniform::new(-10.0, 10.0),
        &mut rng,
    );

    // Scale to reasonable range
    let max_val = embedding.iter()
        .fold(0.0f64, |acc, &x: &f64| acc.max(x.abs()))
        .max(1e-10);
    let scale = 10.0 / max_val;
    embedding.mapv_inplace(|x| x * scale);

    embedding
}

/// PCA dimensionality reduction: project data onto top n_dims principal components.
/// Returns the projected n×n_dims matrix (preserves variance, no std scaling).
pub(crate) fn pca_reduce(data: &Array2<f64>, n_dims: usize) -> Array2<f64> {
    let n_samples = data.nrows();
    let d = data.ncols();

    // Center
    let means: Vec<f64> = (0..d)
        .map(|j| data.column(j).sum() / n_samples as f64)
        .collect();

    let mut centered = data.clone();
    for i in 0..n_samples {
        for j in 0..d {
            centered[[i, j]] -= means[j];
        }
    }

    // Covariance matrix (d×d)
    let mut cov = Array2::<f64>::zeros((d, d));
    for i in 0..n_samples {
        let row = centered.row(i);
        for j in 0..d {
            for k in j..d {
                let v = row[j] * row[k];
                cov[[j, k]] += v;
                if j != k { cov[[k, j]] += v; }
            }
        }
    }
    let inv_n: f64 = 1.0 / (n_samples as f64 - 1.0);
    for j in 0..d {
        for k in 0..d {
            cov[[j, k]] *= inv_n;
        }
    }

    match cov.eigh(UPLO::Lower) {
        Ok((eigenvalues, eigenvectors)) => {
            let eig_vec: Vec<f64> = eigenvalues.to_vec();
            let mut indices: Vec<usize> = (0..d).collect();
            indices.sort_by(|&a, &b| eig_vec[b].partial_cmp(&eig_vec[a]).unwrap());

            // Project onto top n_dims eigenvectors
            let mut result = Array2::<f64>::zeros((n_samples, n_dims));
            for comp in 0..n_dims {
                let eig_idx = indices[comp];
                for i in 0..n_samples {
                    let mut val: f64 = 0.0;
                    for j in 0..d {
                        val += centered[[i, j]] * eigenvectors[[j, eig_idx]];
                    }
                    result[[i, comp]] = val;
                }
            }

            let var_explained: f64 = indices[..n_dims].iter().map(|&i| eig_vec[i]).sum();
            let var_total: f64 = eig_vec.iter().sum();
            eprintln!("PCA: {:.1}% variance explained", 100.0 * var_explained / var_total);

            result
        }
        Err(e) => {
            eprintln!("PCA reduction failed ({}), using original data", e);
            data.clone()
        }
    }
}

/// PCA initialization: project data onto first n_components principal components.
/// Matches uwot's "spca" fallback — scales output to std dev = 1.
/// Computes PCA via the d×d covariance matrix (cheap when d << n).
pub(crate) fn pca_initialization(
    data: &Array2<f64>,
    n_components: usize,
    random_state: Option<u64>,
) -> Array2<f64> {
    let n_samples = data.nrows();
    let n_dims = data.ncols();

    // Center the data
    let means: Vec<f64> = (0..n_dims)
        .map(|j| data.column(j).sum() / n_samples as f64)
        .collect();

    let mut centered = data.clone();
    for i in 0..n_samples {
        for j in 0..n_dims {
            centered[[i, j]] -= means[j];
        }
    }

    // Compute covariance matrix (d×d) — O(n*d²), much cheaper than n×n
    let mut cov = Array2::zeros((n_dims, n_dims));
    for i in 0..n_samples {
        let row = centered.row(i);
        for j in 0..n_dims {
            for k in j..n_dims {
                let v = row[j] * row[k];
                cov[[j, k]] += v;
                if j != k { cov[[k, j]] += v; }
            }
        }
    }
    let inv_n: f64 = 1.0 / (n_samples as f64 - 1.0);
    for j in 0..n_dims {
        for k in 0..n_dims {
            cov[[j, k]] *= inv_n;
        }
    }

    // Eigendecomposition of d×d covariance matrix
    let cov_clone: Array2<f64> = cov;
    match cov_clone.eigh(UPLO::Lower) {
        Ok((eigenvalues, eigenvectors)) => {
            // Eigenvalues are in ascending order, we want the largest
            let eig_vec: Vec<f64> = eigenvalues.to_vec();
            let mut indices: Vec<usize> = (0..n_dims).collect();
            indices.sort_by(|&a, &b| eig_vec[b].partial_cmp(&eig_vec[a]).unwrap());

            // Project onto top n_components eigenvectors
            let mut embedding: Array2<f64> = Array2::zeros((n_samples, n_components));
            for comp in 0..n_components {
                let eig_idx = indices[comp];
                for i in 0..n_samples {
                    let mut val: f64 = 0.0;
                    for j in 0..n_dims {
                        val += centered[[i, j]] * eigenvectors[[j, eig_idx]];
                    }
                    embedding[[i, comp]] = val;
                }
            }

            // Scale to std dev = 1 per component (like uwot's init_sdev=1)
            for comp in 0..n_components {
                let col_mean = embedding.column(comp).sum() / n_samples as f64;
                let var: f64 = embedding.column(comp).iter()
                    .map(|&x| (x - col_mean).powi(2))
                    .sum::<f64>() / (n_samples as f64 - 1.0);
                let std = var.sqrt().max(1e-10);
                for i in 0..n_samples {
                    embedding[[i, comp]] /= std;
                }
            }

            // Add small noise for stability
            let mut rng = match random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::from_entropy(),
            };
            let noise_scale = 0.0001;
            let dist = StandardNormal;
            for i in 0..n_samples {
                for j in 0..n_components {
                    let noise: f64 = dist.sample(&mut rng);
                    embedding[[i, j]] += noise * noise_scale;
                }
            }

            eprintln!("PCA init: top eigenvalues = [{:.4}, {:.4}]",
                      eigenvalues[indices[0]], eigenvalues[indices[1]]);
            embedding
        }
        Err(e) => {
            eprintln!("PCA failed ({}), falling back to random init", e);
            random_initialization(n_samples, n_components, random_state)
        }
    }
}

/// Compute the Laplacian of the graph (for reference, not currently used)
#[allow(dead_code)]
fn compute_graph_laplacian(graph: &Array2<f64>) -> Array2<f64> {
    let n = graph.nrows();
    let mut laplacian = Array2::zeros((n, n));

    // Compute degree matrix
    let degrees: Vec<f64> = (0..n)
        .map(|i| graph.row(i).sum())
        .collect();

    // L = D - A
    for i in 0..n {
        laplacian[[i, i]] = degrees[i];
        for j in 0..n {
            if i != j {
                laplacian[[i, j]] = -graph[[i, j]];
            }
        }
    }

    laplacian
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_spectral_layout() {
        let graph = SparseGraph::from_triplets(5, &[], &[], &[]);
        let embedding = spectral_layout(&graph, 2, Some(42));
        assert_eq!(embedding.shape(), &[5, 2]);
    }

    #[test]
    fn test_graph_laplacian() {
        let graph = Array2::from_shape_vec((3, 3), vec![
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
        ]).unwrap();

        let laplacian = compute_graph_laplacian(&graph);
        assert_eq!(laplacian.shape(), &[3, 3]);

        // Check that diagonal contains degrees
        assert_eq!(laplacian[[0, 0]], 1.0);
        assert_eq!(laplacian[[1, 1]], 2.0);
        assert_eq!(laplacian[[2, 2]], 1.0);
    }
}
