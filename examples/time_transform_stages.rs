//! Where the transform's time goes: kNN + sigma + init (transform_stages) vs the SGD after it.
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
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(40);
    let n_new: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let train = blobs(100_000, d, 12, 42);
    let rest = blobs(n_new, d, 12, 43);
    let (_, model) = umaprs::UMAP::new()
        .n_neighbors(15)
        .min_dist(0.01)
        .n_epochs(200)
        .random_state(42)
        .fit(&train);
    let t = Instant::now();
    let st = model.transform_stages(&rest);
    let t_knn = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = model.transform(&rest);
    let t_all = t.elapsed().as_secs_f64();
    println!(
        "d={d} n_new={n_new}: stages (kNN+sigma+init) {t_knn:.1} s | full transform {t_all:.1} s | so SGD ~{:.1} s | init rows {}",
        t_all - t_knn,
        st.init.nrows()
    );
}
