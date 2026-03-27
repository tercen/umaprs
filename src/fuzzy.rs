use ndarray::{Array2, ArrayView1};
use std::collections::HashMap;

use crate::sparse::SparseGraph;

/// Compute Euclidean distance between two points
fn euclidean_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Smooth k-nearest neighbor distances to create local connectivity
fn smooth_knn_distances(distances: &[f64], target: f64) -> (f64, f64) {
    let rho = distances[0].max(0.0); // Distance to nearest neighbor

    // Binary search for sigma
    let mut lo = 0.0;
    let mut hi = 1e10;
    let mut mid = 1.0;

    for _ in 0..64 {
        mid = (lo + hi) / 2.0;
        let mut val = 0.0;
        for &d in distances.iter().skip(1) {
            val += (-((d - rho).max(0.0) / mid)).exp();
        }

        if (val - target).abs() < 1e-5 {
            break;
        }

        if val > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    (rho, mid)
}

/// Compute the fuzzy simplicial set (graph) representation.
/// Returns a sparse symmetric graph.
/// Result of fuzzy simplicial set computation — includes sigmas/rhos for model export
pub struct FuzzyResult {
    pub graph: SparseGraph,
    pub sigmas: Vec<f64>,
    pub rhos: Vec<f64>,
}

pub fn compute_fuzzy_simplicial_set(
    knn_indices: &Array2<usize>,
    data: &Array2<f64>,
    k: usize,
) -> SparseGraph {
    let result = compute_fuzzy_simplicial_set_full(knn_indices, data, k);
    result.graph
}

pub fn compute_fuzzy_simplicial_set_full(
    knn_indices: &Array2<usize>,
    data: &Array2<f64>,
    k: usize,
) -> FuzzyResult {
    let n_samples = data.nrows();

    // Compute rho and sigma for each point
    let mut rhos = Vec::with_capacity(n_samples);
    let mut sigmas = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let point = data.row(i);
        let neighbors = knn_indices.row(i);

        let mut distances: Vec<f64> = neighbors
            .iter()
            .map(|&j| euclidean_distance(point, data.row(j)))
            .collect();
        distances.insert(0, 0.0); // Distance to self

        let target = (k as f64).ln();
        let (rho, sigma) = smooth_knn_distances(&distances, target);
        rhos.push(rho);
        sigmas.push(sigma);
    }

    // Compute directed membership strengths into a HashMap (sparse)
    let mut directed: HashMap<(usize, usize), f64> = HashMap::new();

    for i in 0..n_samples {
        let point = data.row(i);
        let neighbors = knn_indices.row(i);
        let rho = rhos[i];
        let sigma = sigmas[i];

        for &j in neighbors.iter() {
            let dist = euclidean_distance(point, data.row(j));
            let val = (-((dist - rho).max(0.0) / sigma.max(1e-10))).exp();
            directed.insert((i, j), val);
        }
    }

    // Symmetrize using probabilistic t-conorm: a + b - a*b
    // Only iterate over existing edges, not n²
    let mut symmetric: HashMap<(usize, usize), f64> = HashMap::new();

    for (&(i, j), &a) in &directed {
        if symmetric.contains_key(&(i, j)) {
            continue;
        }
        let b = directed.get(&(j, i)).copied().unwrap_or(0.0);
        let val = a + b - a * b;
        if val > 1e-8 {
            symmetric.insert((i, j), val);
            symmetric.insert((j, i), val);
        }
    }

    // Build SparseGraph from triplets
    let mut rows = Vec::with_capacity(symmetric.len());
    let mut cols = Vec::with_capacity(symmetric.len());
    let mut vals = Vec::with_capacity(symmetric.len());

    for (&(i, j), &v) in &symmetric {
        rows.push(i);
        cols.push(j);
        vals.push(v);
    }

    FuzzyResult {
        graph: SparseGraph::from_triplets(n_samples, &rows, &cols, &vals),
        sigmas,
        rhos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_smooth_knn_distances() {
        let distances = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (rho, sigma) = smooth_knn_distances(&distances, 1.6);
        assert!(rho >= 0.0);
        assert!(sigma > 0.0);
    }

    #[test]
    fn test_fuzzy_simplicial_set() {
        let data = Array2::from_shape_vec((5, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            10.0, 10.0,
            11.0, 10.0,
        ]).unwrap();

        let knn_indices = Array2::from_shape_vec((5, 2), vec![
            1, 2,
            0, 2,
            0, 1,
            4, 3,
            3, 4,
        ]).unwrap();

        let graph = compute_fuzzy_simplicial_set(&knn_indices, &data, 2);
        assert_eq!(graph.n_nodes, 5);
        assert!(graph.nnz() > 0);

        // Check symmetry via dense conversion
        let dense = graph.to_dense();
        for i in 0..5 {
            for j in 0..5 {
                assert!((dense[[i, j]] - dense[[j, i]]).abs() < 1e-10);
            }
        }
    }
}
