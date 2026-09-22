//! The scale case: fit on a 100k-cell training draw, transform 400k more, all at 40 markers.
//! The old transform scanned every training point per query -- O(n_new x n_train).
use ndarray::Array2;
use std::time::Instant;
fn blobs(n: usize, d: usize, nb: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    let mut unit = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut centres = vec![0.0f64; nb * d];
    for c in centres.iter_mut() {
        *c = (unit() - 0.5) * 12.0;
    }
    let mut v = vec![0.0f64; n * d];
    for i in 0..n {
        let b = i % nb;
        for j in 0..d {
            let u1 = unit().max(1e-12);
            let u2 = unit();
            v[i * d + j] = centres[b * d + j]
                + 0.9 * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        }
    }
    Array2::from_shape_vec((n, d), v).unwrap()
}
fn main() {
    // Optional args: n_train n_new (default 100k / 400k)
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|v| v.parse().ok())
        .collect();
    let (n_train, n_new) = (
        a.first().copied().unwrap_or(100_000),
        a.get(1).copied().unwrap_or(400_000),
    );
    let (d, nb) = (40usize, 12usize);
    let train = blobs(n_train, d, nb, 42);
    let rest = blobs(n_new, d, nb, 43);
    let t = Instant::now();
    let (emb, model) = umaprs::UMAP::new()
        .n_neighbors(15)
        .min_dist(0.01)
        .n_epochs(200)
        .random_state(42)
        .fit(&train);
    let t_fit = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let proj = model.transform(&rest);
    let t_tr = t.elapsed().as_secs_f64();
    println!(
        "fit {n_train} x {d}: {t_fit:.1} s | transform {n_new}: {t_tr:.1} s | emb {:?} proj {:?}",
        emb.dim(),
        proj.dim()
    );
}
