use ndarray::Array2;
use umaprs::UMAP;
use std::time::Instant;

fn main() {
    for &(n, n_features) in &[
        (1000, 50), (5000, 50), (10000, 50),
    ] {
        let mut data_vec = Vec::with_capacity(n * n_features);
        for i in 0..n {
            for j in 0..n_features {
                data_vec.push((i * j) as f64 / n as f64 + (i as f64 * 0.1).sin());
            }
        }
        let data = Array2::from_shape_vec((n, n_features), data_vec).unwrap();

        let umap = UMAP::new()
            .n_neighbors(15)
            .n_epochs(200)
            .random_state(42);

        let start = Instant::now();
        let embedding = umap.fit_transform(&data);
        let elapsed = start.elapsed();

        println!(
            "n={:>5}, dims={}, time={:.3}s, shape={:?}",
            n, n_features, elapsed.as_secs_f64(), embedding.shape()
        );
    }
}
