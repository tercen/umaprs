//! The fuzzy simplicial set, as `umap-learn` defines it.
//!
//! Three functions mirror `umap.umap_` one for one — [`smooth_knn_dist`],
//! [`compute_membership_strengths`], [`fuzzy_simplicial_set`] — and are checked against its
//! outputs stage by stage in `tests/umap_learn_parity.rs`, given the same kNN.
//!
//! Conventions that the earlier version got wrong, each of which changes the graph:
//! - **ρ is the distance to the first non-zero neighbour**, not 0. It is UMAP's local
//!   connectivity assumption: the nearest neighbour has membership 1.
//! - **The σ search targets `log2(k)`**, not `ln(k)`. At k = 15 that is 3.9 against 2.7 —
//!   roughly twice the effective neighbourhood.
//! - **σ has a floor**, `MIN_K_DIST_SCALE` times a mean distance. Without it a point whose
//!   neighbours are all exact duplicates — common after asinh — gets σ → 0 and every
//!   non-duplicate neighbour a membership of 0.
//! - **`k` counts the point itself.** The kNN rows carry self at column 0 with distance 0, as
//!   `umap-learn`'s do, so `n_neighbors = 15` means fourteen other points.
//!
//! The symmetrisation is the fuzzy union `A + Aᵀ − A∘Aᵀ`, done as a merge of two CSR
//! matrices: parallel per row, and deterministic by construction rather than by a later sort.
use ndarray::Array2;
use rayon::prelude::*;

use crate::sparse::SparseGraph;

/// `umap.umap_.SMOOTH_K_TOLERANCE`
pub const SMOOTH_K_TOLERANCE: f64 = 1e-5;
/// `umap.umap_.MIN_K_DIST_SCALE`
pub const MIN_K_DIST_SCALE: f64 = 1e-3;
const N_ITER: usize = 64;

/// Per-point `(sigma, rho)` — `smooth_knn_dist`.
///
/// `dists` is `n × k`, row-major, column 0 the point itself at distance 0 (or, for the
/// transform's queries, the nearest training point — the loop skips column 0 either way, which
/// is `umap-learn`'s behaviour and is reproduced deliberately).
pub fn smooth_knn_dist(
    dists: &[f64],
    n: usize,
    k: usize,
    local_connectivity: f64,
) -> (Vec<f64>, Vec<f64>) {
    let target = (k as f64).log2();
    let mean_distances = dists.iter().sum::<f64>() / dists.len().max(1) as f64;

    let (sigmas, rhos): (Vec<f64>, Vec<f64>) = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &dists[i * k..(i + 1) * k];
            let non_zero: Vec<f64> = row.iter().copied().filter(|d| *d > 0.0).collect();
            let mut rho = 0.0;
            if non_zero.len() as f64 >= local_connectivity {
                let index = local_connectivity.floor() as usize;
                let interpolation = local_connectivity - index as f64;
                if index > 0 {
                    rho = non_zero[index - 1];
                    if interpolation > SMOOTH_K_TOLERANCE {
                        rho += interpolation * (non_zero[index] - non_zero[index - 1]);
                    }
                } else {
                    rho = interpolation * non_zero[0];
                }
            } else if !non_zero.is_empty() {
                rho = non_zero.iter().cloned().fold(f64::MIN, f64::max);
            }

            let (mut lo, mut hi, mut mid) = (0.0f64, f64::INFINITY, 1.0f64);
            for _ in 0..N_ITER {
                let mut psum = 0.0;
                for &dj in &row[1..] {
                    let d = dj - rho;
                    psum += if d > 0.0 { (-d / mid).exp() } else { 1.0 };
                }
                if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                    break;
                }
                if psum > target {
                    hi = mid;
                    mid = (lo + hi) / 2.0;
                } else {
                    lo = mid;
                    mid = if hi.is_infinite() {
                        mid * 2.0
                    } else {
                        (lo + hi) / 2.0
                    };
                }
            }
            let mut sigma = mid;
            if rho > 0.0 {
                let mean_ith = row.iter().sum::<f64>() / k as f64;
                if sigma < MIN_K_DIST_SCALE * mean_ith {
                    sigma = MIN_K_DIST_SCALE * mean_ith;
                }
            } else if sigma < MIN_K_DIST_SCALE * mean_distances {
                sigma = MIN_K_DIST_SCALE * mean_distances;
            }
            (sigma, rho)
        })
        .unzip();
    (sigmas, rhos)
}

/// Directed memberships — `compute_membership_strengths`. Returns `(rows, cols, vals)` with one
/// entry per kNN slot; the self edge carries 0 and is dropped by the caller.
pub fn compute_membership_strengths(
    inds: &[usize],
    dists: &[f64],
    n: usize,
    k: usize,
    sigmas: &[f64],
    rhos: &[f64],
    bipartite: bool,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rows = Vec::with_capacity(n * k);
    let mut cols = Vec::with_capacity(n * k);
    let mut vals = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            let col = inds[i * k + j];
            let val = if !bipartite && col == i {
                0.0
            } else if dists[i * k + j] - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-(dists[i * k + j] - rhos[i]) / sigmas[i]).exp()
            };
            rows.push(i);
            cols.push(col);
            vals.push(val);
        }
    }
    (rows, cols, vals)
}

/// What the fit keeps: the symmetric graph and the per-point σ, ρ the model needs later.
pub struct FuzzyResult {
    pub graph: SparseGraph,
    pub sigmas: Vec<f64>,
    pub rhos: Vec<f64>,
}

/// `fuzzy_simplicial_set` with `set_op_mix_ratio = 1`: the fuzzy union `A + Aᵀ − A∘Aᵀ`.
///
/// `inds`/`dists` are `n × k` with self at column 0.
pub fn fuzzy_simplicial_set(inds: &[usize], dists: &[f64], n: usize, k: usize) -> FuzzyResult {
    let (sigmas, rhos) = smooth_knn_dist(dists, n, k, 1.0);
    let (rows, cols, vals) = compute_membership_strengths(inds, dists, n, k, &sigmas, &rhos, false);

    // Directed CSR A, rows sorted by column, zeros dropped.
    let a = csr_from_rows(n, &rows, &cols, &vals);
    // Aᵀ in CSR, by counting sort — O(nnz), no hashing.
    let at = transpose(&a);

    // Union, one row at a time, as a merge of two sorted lists.
    let merged: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (ca, va) = a.row_entries(i);
            let (cb, vb) = at.row_entries(i);
            let mut out = Vec::with_capacity(ca.len() + cb.len());
            let (mut p, mut q) = (0, 0);
            while p < ca.len() || q < cb.len() {
                let (col, val) = if q >= cb.len() || (p < ca.len() && ca[p] < cb[q]) {
                    let r = (ca[p], va[p]);
                    p += 1;
                    r
                } else if p >= ca.len() || cb[q] < ca[p] {
                    let r = (cb[q], vb[q]);
                    q += 1;
                    r
                } else {
                    let r = (ca[p], va[p] + vb[q] - va[p] * vb[q]);
                    p += 1;
                    q += 1;
                    r
                };
                if val != 0.0 {
                    out.push((col, val));
                }
            }
            out
        })
        .collect();

    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);
    for r in &merged {
        for &(c, v) in r {
            col_indices.push(c);
            values.push(v);
        }
        row_offsets.push(col_indices.len());
    }
    let graph = SparseGraph {
        n_nodes: n,
        row_offsets,
        col_indices,
        values,
    };
    FuzzyResult {
        graph,
        sigmas,
        rhos,
    }
}

/// CSR from per-row triplets already grouped by row: sort each row by column, drop zeros.
fn csr_from_rows(n: usize, rows: &[usize], cols: &[usize], vals: &[f64]) -> SparseGraph {
    let mut per_row: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for ((&r, &c), &v) in rows.iter().zip(cols).zip(vals) {
        if v != 0.0 {
            per_row[r].push((c, v));
        }
    }
    per_row
        .par_iter_mut()
        .for_each(|r| r.sort_unstable_by_key(|e| e.0));
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);
    for r in &per_row {
        for &(c, v) in r {
            col_indices.push(c);
            values.push(v);
        }
        row_offsets.push(col_indices.len());
    }
    SparseGraph {
        n_nodes: n,
        row_offsets,
        col_indices,
        values,
    }
}

fn transpose(a: &SparseGraph) -> SparseGraph {
    let n = a.n_nodes;
    let mut counts = vec![0usize; n + 1];
    for &c in &a.col_indices {
        counts[c + 1] += 1;
    }
    for i in 1..=n {
        counts[i] += counts[i - 1];
    }
    let mut next = counts.clone();
    let mut col_indices = vec![0usize; a.nnz()];
    let mut values = vec![0.0f64; a.nnz()];
    for (row, col, val) in a.edges() {
        let slot = next[col];
        col_indices[slot] = row;
        values[slot] = val;
        next[col] += 1;
    }
    // Rows of a CSR are emitted in ascending row order, so each transposed row is already sorted.
    SparseGraph {
        n_nodes: n,
        row_offsets: counts,
        col_indices,
        values,
    }
}

// ---- the entry points `lib.rs` calls -----------------------------------------------------------

/// From kNN *indices of other points* (`n × (k−1)`) and the data: distances, self prepended,
/// then the graph. `k` is `n_neighbors`, counting the point itself.
pub fn compute_fuzzy_simplicial_set_full(
    knn_indices: &Array2<usize>,
    data: &Array2<f64>,
    k: usize,
) -> FuzzyResult {
    let n = data.nrows();
    let others = knn_indices.ncols();
    assert_eq!(
        others + 1,
        k,
        "kNN must hold n_neighbors - 1 other points; self is added here"
    );
    let mut inds = vec![0usize; n * k];
    let mut dists = vec![0.0f64; n * k];
    inds.par_chunks_mut(k)
        .zip(dists.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, (ir, dr))| {
            ir[0] = i;
            dr[0] = 0.0;
            let p = data.row(i);
            for j in 0..others {
                let nb = knn_indices[[i, j]];
                ir[j + 1] = nb;
                dr[j + 1] = p
                    .iter()
                    .zip(data.row(nb).iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
            }
        });
    fuzzy_simplicial_set(&inds, &dists, n, k)
}

pub fn compute_fuzzy_simplicial_set(
    knn_indices: &Array2<usize>,
    data: &Array2<f64>,
    k: usize,
) -> SparseGraph {
    compute_fuzzy_simplicial_set_full(knn_indices, data, k).graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_is_the_first_nonzero_neighbour_and_sigma_hits_log2k() {
        // one point, k = 4: self, then three neighbours at 1, 2, 3
        let d = [0.0, 1.0, 2.0, 3.0];
        let (s, r) = smooth_knn_dist(&d, 1, 4, 1.0);
        assert_eq!(r[0], 1.0);
        let psum: f64 = d[1..]
            .iter()
            .map(|x| ((-(x - r[0]).max(0.0)) / s[0]).exp())
            .sum();
        assert!(
            (psum - 2.0f64).abs() < 1e-4,
            "sum {psum} should be log2(4) = 2"
        );
    }

    #[test]
    fn duplicates_get_the_sigma_floor_not_zero() {
        let d = [0.0, 0.0, 0.0, 0.0, 5.0];
        let (s, r) = smooth_knn_dist(&d, 1, 5, 1.0);
        assert_eq!(r[0], 5.0, "the only non-zero distance");
        assert!(s[0] > 0.0);
    }
}
