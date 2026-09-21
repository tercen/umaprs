use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

use crate::hnsw::Hnsw;
use crate::kdtree::KdTree;

/// Threshold for switching from brute-force to tree-based methods
const TREE_THRESHOLD: usize = 500;

/// Max dimensions for kd-tree (above this, HNSW is better)
/// kd-trees stop pruning usefully somewhere around 15-20 dimensions; measured here at 40 dims
/// a query cost 1.4 ms -- close to a full scan -- and the transform of 100k points took 139 s
/// of which the SGD was 0.3 s. Above this, HNSW with an exact 2k refine.
const KDTREE_MAX_DIMS: usize = 16;

/// Compute k-nearest neighbors for each point.
/// Strategy:
///   - Small datasets (<=500): exact brute-force
///   - Large + low-dim (<=40): kd-tree (exact, like uwot's FNN)
///   - Large + high-dim (>40): HNSW (approximate)
pub fn compute_knn_graph(data: &Array2<f64>, k: usize, seed: u64) -> Array2<usize> {
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
        compute_knn_hnsw_f32(data, k, seed)
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

/// Plain HNSW with f32 distances (no quantization)
pub fn compute_knn_hnsw_f32(data: &Array2<f64>, k: usize, seed: u64) -> Array2<usize> {
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

    let hnsw = Hnsw::build(n_samples, &dist_fn, seed);

    // Get 2k candidates from HNSW, refine with exact f64 distances
    let refine_k = (k * 2).min(n_samples - 1);
    let mut knn_indices = Array2::zeros((n_samples, k));
    knn_indices
        .outer_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(i, mut row)| {
            let results = hnsw.search(i as u32, refine_k + 1, &dist_fn);

            let candidates: Vec<usize> = results
                .iter()
                .map(|&(nb, _)| nb as usize)
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
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0, 11.0, 10.0],
        )
        .unwrap();
        let knn = compute_knn_bruteforce(&data, 2);
        assert_eq!(knn.shape(), &[5, 2]);
        assert!(knn[[0, 0]] == 1 || knn[[0, 0]] == 2);
        assert_eq!(knn[[3, 0]], 4);
    }
}

/// k nearest *training* points for each query row — the transform's kNN.
///
/// Exact kd-tree up to `KDTREE_MAX_DIMS`; above that HNSW with a 2k exact refine, as the fit
/// does. The queries are not in the index, so nothing is excluded and no self row is added:
/// this is what `umap-learn`'s `_knn_search_index.query` returns. Distances are exact f64.
/// Parallel over queries.
pub fn compute_knn_external(
    train: &Array2<f64>,
    queries: &Array2<f64>,
    k: usize,
    seed: u64,
) -> (Array2<usize>, Array2<f64>) {
    let (n_train, d) = (train.nrows(), train.ncols());
    let n_q = queries.nrows();
    assert_eq!(
        queries.ncols(),
        d,
        "query dimensionality must match the training data"
    );
    let k = k.min(n_train);
    let flat: Vec<f32> = train.iter().map(|&v| v as f32).collect();
    let mut inds = Array2::zeros((n_q, k));
    let mut dists = Array2::zeros((n_q, k));

    let exact = |qi: usize, cands: &[u32], out_i: &mut [usize], out_d: &mut [f64]| {
        let q = queries.row(qi);
        let mut ex: Vec<(usize, f64)> = cands
            .iter()
            .map(|&j| (j as usize, euclidean_distance(q, train.row(j as usize))))
            .collect();
        ex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        for (slot, &(j, dd)) in ex.iter().take(k).enumerate() {
            out_i[slot] = j;
            out_d[slot] = dd;
        }
    };

    if n_train <= TREE_THRESHOLD {
        inds.outer_iter_mut()
            .zip(dists.outer_iter_mut())
            .enumerate()
            .par_bridge()
            .for_each(|(qi, (mut ri, mut rd))| {
                let all: Vec<u32> = (0..n_train as u32).collect();
                exact(
                    qi,
                    &all,
                    ri.as_slice_mut().unwrap(),
                    rd.as_slice_mut().unwrap(),
                );
            });
    } else if d <= KDTREE_MAX_DIMS {
        let tree = KdTree::build(&flat, n_train, d);
        inds.outer_iter_mut()
            .zip(dists.outer_iter_mut())
            .enumerate()
            .par_bridge()
            .for_each(|(qi, (mut ri, mut rd))| {
                let q: Vec<f32> = queries.row(qi).iter().map(|&v| v as f32).collect();
                let cands: Vec<u32> = tree
                    .query(&q, k.min(n_train))
                    .into_iter()
                    .map(|(i, _)| i)
                    .collect();
                exact(
                    qi,
                    &cands,
                    ri.as_slice_mut().unwrap(),
                    rd.as_slice_mut().unwrap(),
                );
            });
    } else {
        let flat_ref = &flat;
        let dist_fn = |i: u32, j: u32| -> f32 {
            let a = &flat_ref[i as usize * d..(i as usize + 1) * d];
            let b = &flat_ref[j as usize * d..(j as usize + 1) * d];
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>()
        };
        let hnsw = Hnsw::build(n_train, &dist_fn, seed);
        let refine_k = (k * 2).min(n_train);
        inds.outer_iter_mut()
            .zip(dists.outer_iter_mut())
            .enumerate()
            .par_bridge()
            .for_each(|(qi, (mut ri, mut rd))| {
                let q: Vec<f32> = queries.row(qi).iter().map(|&v| v as f32).collect();
                // A query that is not in the graph is addressed as a virtual id: the search only
                // ever asks for the distance from a graph node to the target, so the closure
                // answers with the query vector when it sees that id.
                let virt = n_train as u32;
                let qdist = |i: u32, j: u32| -> f32 {
                    if j == virt {
                        let a = &flat_ref[i as usize * d..(i as usize + 1) * d];
                        a.iter()
                            .zip(&q)
                            .map(|(x, y)| (x - y) * (x - y))
                            .sum::<f32>()
                    } else {
                        dist_fn(i, j)
                    }
                };
                let cands: Vec<u32> = hnsw
                    .search(virt, refine_k, &qdist)
                    .into_iter()
                    .map(|(i, _)| i)
                    .collect();
                exact(
                    qi,
                    &cands,
                    ri.as_slice_mut().unwrap(),
                    rd.as_slice_mut().unwrap(),
                );
            });
    }
    (inds, dists)
}
