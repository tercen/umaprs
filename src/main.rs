use ndarray::Array2;
use umaprs::UMAP;

fn main() {
    println!("UMAP Example");
    println!("============\n");

    // Create sample data: 100 points in 10-dimensional space
    let n_samples = 100;
    let n_features = 10;
    let mut data_vec = Vec::with_capacity(n_samples * n_features);

    for i in 0..n_samples {
        for j in 0..n_features {
            data_vec.push((i * j) as f64 + (i as f64).sin());
        }
    }

    let data = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .expect("Failed to create data array");

    println!("Input data shape: {:?}", data.shape());

    // Create UMAP instance with custom parameters
    let umap = UMAP::new()
        .n_neighbors(15)
        .n_components(2)
        .min_dist(0.1)
        .learning_rate(1.0)
        .n_epochs(100)
        .random_state(42);

    println!("Running UMAP dimensionality reduction...");

    // Fit and transform the data
    let embedding = umap.fit_transform(&data);

    println!("Output embedding shape: {:?}", embedding.shape());
    println!("\nFirst 5 embedded points:");
    for i in 0..5.min(embedding.nrows()) {
        println!("  Point {}: [{:.3}, {:.3}]", i, embedding[[i, 0]], embedding[[i, 1]]);
    }

    println!("\nUMAP dimensionality reduction completed!");
}
