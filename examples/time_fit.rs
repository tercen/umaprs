//! Wall-clock of one fit on Levine-sized synthetic data (50k cells, 32 markers, 12 blobs).
use ndarray::Array2;
use std::time::Instant;
fn main() {
    let (n, d, nb) = (50_000usize, 32usize, 12usize);
    // deterministic LCG so both trees see identical data
    let mut s: u64 = 42;
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
    let mut data = vec![0.0f64; n * d];
    for i in 0..n {
        let b = i % nb;
        for j in 0..d {
            let u1 = unit().max(1e-12);
            let u2 = unit();
            let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            data[i * d + j] = centres[b * d + j] + 0.9 * g;
        }
    }
    let data = Array2::from_shape_vec((n, d), data).unwrap();
    let t = Instant::now();
    let emb = umaprs::UMAP::new()
        .n_neighbors(15)
        .min_dist(0.01)
        .n_epochs(200)
        .random_state(42)
        .fit_transform(&data);
    let secs = t.elapsed().as_secs_f64();
    let (mut sx, mut sy) = (0.0, 0.0);
    for i in 0..n {
        sx += emb[[i, 0]].abs();
        sy += emb[[i, 1]].abs();
    }
    println!(
        "fit 50k x 32: {secs:.2} s  (mean |x| {:.3}, mean |y| {:.3})",
        sx / n as f64,
        sy / n as f64
    );
}
