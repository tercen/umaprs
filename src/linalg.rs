//! Symmetric eigendecomposition in pure Rust.
//!
//! This used to be LAPACK through `ndarray-linalg` with a statically built OpenBLAS. OpenBLAS
//! compiles its kernels for the CPU of the machine that builds it, so a CI runner with AVX-512
//! failed to build it (gcc 12 / OpenBLAS flag mismatch on `ssum_k`) and a runner without it
//! produced a binary tuned to that runner — neither is acceptable for a shipped image. The only
//! consumer is a dense symmetric eigenproblem of at most a few thousand rows (the spectral start,
//! capped at `SPECTRAL_DENSE_THRESHOLD`) or `d × d` covariances (PCA), which nalgebra's
//! tridiagonal QR handles in the same O(n³).
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

/// Eigenvalues in ascending order and the matching eigenvectors as columns, as LAPACK's `dsyev`
/// returns them. `a` must be symmetric; only its values are read.
pub fn eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(format!("eigh: matrix is {} x {}, not square", n, a.ncols()));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err("eigh: matrix has a non-finite entry".to_string());
    }
    let m = DMatrix::from_fn(n, n, |i, j| a[[i, j]]);
    let se = m.clone().symmetric_eigen();
    // nalgebra 0.33 can return a vector paired with a *neighbouring* eigenvalue when two are
    // close (seen at 0 vs 1.9e-5 on a clustered graph's Laplacian: the trivial vector came back
    // as the second eigenvector). Each vector's own Rayleigh quotient is its eigenvalue; order
    // and report by that.
    let mu = &m * &se.eigenvectors;
    let rq: Vec<f64> = (0..n)
        .map(|c| se.eigenvectors.column(c).dot(&mu.column(c)))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| {
        rq[x]
            .partial_cmp(&rq[y])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let values = Array1::from_iter(order.iter().map(|&k| rq[k]));
    let vectors = Array2::from_shape_fn((n, n), |(i, c)| se.eigenvectors[(i, order[c])]);
    Ok((values, vectors))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_a_symmetric_matrix_and_orders_ascending() {
        let a = ndarray::arr2(&[[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]]);
        let (w, v) = eigh(&a).unwrap();
        assert!(w[0] <= w[1] && w[1] <= w[2]);
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3).map(|k| v[[i, k]] * w[k] * v[[j, k]]).sum();
                assert!(
                    (r - a[[i, j]]).abs() < 1e-12,
                    "({i},{j}) {r} vs {}",
                    a[[i, j]]
                );
            }
        }
    }
}

#[cfg(test)]
mod pairing {
    use super::*;

    /// Each returned value is the Rayleigh quotient of its own vector, in ascending order, on
    /// the Laplacian of a clustered graph where nalgebra's own pairing slips.
    #[test]
    fn values_belong_to_their_vectors() {
        let g = crate::sparse_spectral::tests::ring_graph(6, 60);
        let n = g.n_nodes;
        let a = g.to_dense();
        let d: Vec<f64> = (0..n).map(|i| a.row(i).sum()).collect();
        let l = Array2::from_shape_fn((n, n), |(i, j)| {
            (if i == j { 1.0 } else { 0.0 }) - a[[i, j]] / (d[i] * d[j]).sqrt()
        });
        let (vals, vecs) = eigh(&l).unwrap();
        for c in 0..n {
            let u = vecs.column(c);
            let rq = u.dot(&l.dot(&u));
            assert!(
                (rq - vals[c]).abs() < 1e-10,
                "column {c}: value {} rq {rq}",
                vals[c]
            );
        }
        assert!(vals.windows(2).into_iter().all(|w| w[0] <= w[1]));
        assert!(vals[0].abs() < 1e-10, "the trivial eigenvalue comes first");
    }
}
