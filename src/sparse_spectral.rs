//! Spectral start for graphs too large for a dense eigendecomposition: uwot's `spectral` init.
//!
//! uwot (`spectral_init` → `rspectra_normalized_laplacian_init`) takes the eigenvectors of the
//! normalised Laplacian `L = I − D^{-1/2} A D^{-1/2}` for the second and third smallest
//! eigenvalues (RSpectra `eigs_sym(L, k = ndim + 1, which = "SM", tol = 1e-4)`), scales them so the
//! largest coordinate is 10 and adds N(0, 1e-4) noise (`scale_and_jitter`). A graph with more than
//! one connected component gets uwot's `spca` instead; so does a solve that does not converge.
//!
//! Here the same vectors come from the top of `M = D^{-1/2} A D^{-1/2}` (`L = I − M`), with the
//! trivial eigenvector `D^{1/2} 1` (eigenvalue 1 on a connected graph) projected out, by a
//! thick-restart Lanczos iteration with full reorthogonalisation (Wu & Simon 2000). Every dot
//! product is summed in one fixed order and the matrix-vector product is per row, so the result
//! depends on the seed only, not on the thread count.

use ndarray::Array2;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::linalg::eigh;
use crate::sparse::SparseGraph;

/// Why the sparse spectral start was not used.
#[derive(Debug, Clone, PartialEq)]
pub enum SpectralFallback {
    /// More than one connected component (uwot then uses `spca`).
    Disconnected(usize),
    /// The eigensolver did not reach the tolerance within its budget.
    NotConverged,
}

const TOL: f64 = 1e-4;
const MAX_MATVECS: usize = 20_000;
/// ARPACK's floor for the relative test: eps^(2/3).
const EPS23: f64 = 3.7e-11;

/// The eigenvectors as uwot's spectral init returns them, scaled and jittered, or why not.
pub fn sparse_spectral_layout(
    graph: &SparseGraph,
    n_components: usize,
    seed: u64,
) -> Result<Array2<f64>, SpectralFallback> {
    let n = graph.n_nodes;
    let comps = connected_components(graph);
    if comps > 1 {
        return Err(SpectralFallback::Disconnected(comps));
    }
    let degree: Vec<f64> = (0..n)
        .map(|i| graph.row_entries(i).1.iter().sum::<f64>())
        .collect();
    let dinv: Vec<f64> = degree.iter().map(|&d| 1.0 / d.sqrt()).collect();
    let mut u1: Vec<f64> = degree.iter().map(|&d| d.sqrt()).collect();
    normalise(&mut u1);

    let op = |x: &[f64], y: &mut [f64]| {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let (cols, vals) = graph.row_entries(i);
            let mut s = 0.0;
            for (&j, &a) in cols.iter().zip(vals) {
                s += a * dinv[j] * x[j];
            }
            *yi = dinv[i] * s;
        });
        deflate(y, &u1);
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut v0: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    deflate(&mut v0, &u1);
    normalise(&mut v0);

    let vecs = lanczos_top(n, n_components, &op, v0).ok_or(SpectralFallback::NotConverged)?;

    // uwot: column 1 is the eigenvector of the second-smallest L eigenvalue (largest here).
    let mut emb = Array2::zeros((n, n_components));
    for (c, v) in vecs.iter().enumerate() {
        // An eigenvector's sign is arbitrary; fix it so the result is reproducible.
        let (imax, _) =
            v.iter().enumerate().fold(
                (0, 0.0f64),
                |b, (i, &x)| if x.abs() > b.1 { (i, x.abs()) } else { b },
            );
        let sign = if v[imax] < 0.0 { -1.0 } else { 1.0 };
        for i in 0..n {
            emb[[i, c]] = sign * v[i];
        }
    }
    // scale_and_jitter(coords, max_coord = 10, sd = 1e-4)
    let max_abs = emb.iter().fold(0.0f64, |m, &x| m.max(x.abs())).max(1e-300);
    let expansion = 10.0 / max_abs;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5bec_7a1e);
    emb.mapv_inplace(|x| x * expansion);
    for x in emb.iter_mut() {
        let z: f64 = StandardNormal.sample(&mut rng);
        *x += 1e-4 * z;
    }
    Ok(emb)
}

/// Number of connected components (union-find over the edges).
pub fn connected_components(graph: &SparseGraph) -> usize {
    let n = graph.n_nodes;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for i in 0..n {
        let (cols, vals) = graph.row_entries(i);
        for (&j, &w) in cols.iter().zip(vals) {
            if w > 0.0 {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
    }
    (0..n).filter(|&i| find(&mut parent, i) == i).count()
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalise(v: &mut [f64]) {
    let s = dot(v, v).sqrt();
    if s > 0.0 {
        v.iter_mut().for_each(|x| *x /= s);
    }
}

fn deflate(v: &mut [f64], u: &[f64]) {
    let c = dot(v, u);
    v.iter_mut().zip(u).for_each(|(x, &ui)| *x -= c * ui);
}

/// Top `nev` eigenvectors (largest eigenvalue first) of a symmetric operator, by thick-restart
/// Lanczos. `None` if the residuals do not fall below `TOL` within `MAX_MATVECS` products.
#[allow(clippy::needless_range_loop)] // T is filled by index on both triangles
fn lanczos_top(
    n: usize,
    nev: usize,
    op: &dyn Fn(&[f64], &mut [f64]),
    v0: Vec<f64>,
) -> Option<Vec<Vec<f64>>> {
    let m = (2 * nev + 30).min(n.saturating_sub(1)).max(nev + 1);
    let keep = (nev + 10).min(m - 1);
    let mut basis: Vec<Vec<f64>> = vec![v0];
    // T = Vᵀ M V, dense (thick restart makes it arrowhead, not tridiagonal).
    let mut t = vec![vec![0.0f64; m]; m];
    let mut w = vec![0.0f64; n];
    let mut matvecs = 0;
    let mut start = 0;
    loop {
        let mut beta = 0.0;
        for j in start..m {
            op(&basis[j], &mut w);
            matvecs += 1;
            // Full Gram-Schmidt, twice: the projections are the column of T.
            for pass in 0..2 {
                for (i, b) in basis.iter().enumerate() {
                    let h = dot(b, &w);
                    if pass == 0 {
                        t[i][j] = h;
                    } else {
                        t[i][j] += h;
                    }
                    w.iter_mut().zip(b).for_each(|(x, &bi)| *x -= h * bi);
                }
            }
            for i in 0..j {
                t[j][i] = t[i][j];
            }
            beta = dot(&w, &w).sqrt();
            if j + 1 < m {
                if beta < 1e-14 {
                    break; // invariant subspace: the Ritz values are exact
                }
                basis.push(w.iter().map(|x| x / beta).collect());
            }
        }
        let k = basis.len();
        let tk = Array2::from_shape_fn((k, k), |(a, b)| t[a][b]);
        let (vals, vecs) = eigh(&tk).ok()?;
        // Largest first.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
        let converged = order[..nev]
            .iter()
            // RSpectra's test on L: residual below tol × |eigenvalue of L| (= 1 − θ here), so
            // the near-1 eigenvalues of clustered graphs, ~1e-4 apart, are actually resolved.
            .all(|&c| (beta * vecs[[k - 1, c]]).abs() < TOL * (1.0 - vals[c]).abs().max(EPS23));
        let ritz = |c: usize| -> Vec<f64> {
            let mut y = vec![0.0; n];
            for (i, b) in basis.iter().enumerate() {
                let s = vecs[[i, c]];
                y.iter_mut().zip(b).for_each(|(x, &bi)| *x += s * bi);
            }
            y
        };
        if converged || beta < 1e-14 {
            return Some(order[..nev].iter().map(|&c| ritz(c)).collect());
        }
        if matvecs >= MAX_MATVECS {
            return None;
        }
        // Thick restart: keep the `keep` best Ritz vectors, continue from the residual.
        let p = keep.min(k - 1);
        let kept: Vec<Vec<f64>> = order[..p].iter().map(|&c| ritz(c)).collect();
        for row in t.iter_mut() {
            row.iter_mut().for_each(|x| *x = 0.0);
        }
        for (i, &c) in order[..p].iter().enumerate() {
            t[i][i] = vals[c];
        }
        basis = kept;
        basis.push(w.iter().map(|x| x / beta).collect());
        start = p;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `k` dense clusters of unequal size in a chain with unequal bridges: connected, with clear
    /// low Laplacian modes and no symmetry (an exactly repeated eigenvalue has a one-dimensional
    /// trace in any single-vector Krylov space -- ARPACK's limit too -- and real kNN graphs
    /// never have one).
    pub(crate) fn ring_graph(k: usize, per: usize) -> SparseGraph {
        let sizes: Vec<usize> = (0..k).map(|g| per + 17 * g).collect();
        let starts: Vec<usize> = sizes
            .iter()
            .scan(0, |acc, &s| {
                let st = *acc;
                *acc += s;
                Some(st)
            })
            .collect();
        let n: usize = sizes.iter().sum();
        let (mut r, mut c, mut v) = (vec![], vec![], vec![]);
        let mut add = |a: usize, b: usize, w: f64| {
            r.push(a);
            c.push(b);
            v.push(w);
            r.push(b);
            c.push(a);
            v.push(w);
        };
        for g in 0..k {
            let (st, sz) = (starts[g], sizes[g]);
            for i in 0..sz {
                for j in (i + 1)..sz.min(i + 6) {
                    add(st + i, st + j, 1.0);
                }
            }
            if g + 1 < k {
                add(st, starts[g + 1] + 1, 0.03 + 0.02 * g as f64);
            }
        }
        SparseGraph::from_triplets(n, &r, &c, &v)
    }

    /// Same subspace as the dense eigendecomposition of L on a graph small enough for both.
    /// The dense vectors are identified by their Rayleigh quotient, not by the solver's value
    /// order: nalgebra's `symmetric_eigen` can pair a vector with a neighbouring eigenvalue when
    /// the small eigenvalues are ~1e-5 apart, as on clustered graphs.
    #[test]
    fn matches_the_dense_eigenvectors() {
        let g = ring_graph(6, 60);
        let n = g.n_nodes;
        let emb = sparse_spectral_layout(&g, 2, 1).expect("connected and converges");
        let a = g.to_dense();
        let d: Vec<f64> = (0..n).map(|i| a.row(i).sum()).collect();
        let l = Array2::from_shape_fn((n, n), |(i, j)| {
            (if i == j { 1.0 } else { 0.0 }) - a[[i, j]] / (d[i] * d[j]).sqrt()
        });
        let lmul = |x: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|i| (0..n).map(|j| l[[i, j]] * x[j]).sum())
                .collect()
        };
        let (_, vecs) = eigh(&l).unwrap();
        let mut rq: Vec<(f64, usize)> = (0..n)
            .map(|c| {
                let u: Vec<f64> = vecs.column(c).to_vec();
                (dot(&u, &lmul(&u)), c)
            })
            .collect();
        rq.sort_by(|x, y| x.0.total_cmp(&y.0));
        for c in 0..2 {
            let mut x: Vec<f64> = emb.column(c).to_vec();
            normalise(&mut x);
            // Rayleigh quotient equals the dense eigenvalue it should be (2nd, 3rd smallest).
            let want = rq[c + 1].0;
            let got = dot(&x, &lmul(&x));
            assert!(
                (got - want).abs() < 1e-3 * want,
                "column {c}: {got:e} vs {want:e}"
            );
            // And it lies in the dense 2-D subspace.
            let mut resid = x.clone();
            for &(_, e) in &rq[1..3] {
                let u: Vec<f64> = vecs.column(e).to_vec();
                let h = dot(&x, &u);
                resid.iter_mut().zip(&u).for_each(|(r, &ui)| *r -= h * ui);
            }
            let r = dot(&resid, &resid).sqrt();
            assert!(r < 1e-3, "column {c} leaves the dense subspace by {r:e}");
        }
    }

    #[test]
    fn scaled_like_uwot_and_reproducible() {
        let g = ring_graph(5, 500);
        let a = sparse_spectral_layout(&g, 2, 7).unwrap();
        let max = a.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        assert!(
            (max - 10.0).abs() < 1e-3,
            "largest coordinate 10, got {max}"
        );
        assert_eq!(a, sparse_spectral_layout(&g, 2, 7).unwrap());
    }

    #[test]
    fn disconnected_graph_falls_back() {
        let g = SparseGraph::from_triplets(4, &[0, 1, 2, 3], &[1, 0, 3, 2], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(connected_components(&g), 2);
        assert_eq!(
            sparse_spectral_layout(&g, 2, 1),
            Err(SpectralFallback::Disconnected(2))
        );
    }
}
