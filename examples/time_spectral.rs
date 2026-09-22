use ndarray::Array2;
use rand::{Rng, SeedableRng};
fn main() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    let data = Array2::from_shape_fn((2000, 38), |_| rng.gen_range(-1.0f64..1.0));
    let t = std::time::Instant::now();
    let _ = umaprs::UMAP::new()
        .n_neighbors(15)
        .init(umaprs::InitMethod::Spectral)
        .threads(1)
        .fit_transform(&data);
    println!(
        "spectral fit 2000 x 38 at threads(1): {:.1} s",
        t.elapsed().as_secs_f64()
    );
}
