use ndarray::Array2;
use umaprs::UMAP;
use std::fs::File;
use std::io::{Write, BufReader, BufRead};

fn main() {
    println!("=== Running UMAP on Crabs Dataset (Rust implementation) ===\n");

    // Read crabs data from CSV
    let file = File::open("data/crabs_data.csv").expect("Failed to open data/crabs_data.csv");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header
    lines.next();

    let mut data_vec = Vec::new();
    let mut n_samples = 0;

    for line in lines {
        let line = line.expect("Failed to read line");
        let values: Vec<f64> = line.split(',')
            .map(|s| s.trim().parse::<f64>().expect("Failed to parse number"))
            .collect();
        data_vec.extend(values);
        n_samples += 1;
    }

    let n_features = data_vec.len() / n_samples;

    let data = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .expect("Failed to create data array");

    println!("Input data shape: {:?}", data.shape());
    println!("Features: FL, RW, CL, CW, BD (5 morphological measurements)");
    println!("Samples: {} crabs", n_samples);

    // Run UMAP with standard parameters
    let umap = UMAP::new()
        .n_neighbors(15)
        .n_components(2)
        .min_dist(0.1)
        .learning_rate(1.0)
        .n_epochs(200)  // Match uwot
        .random_state(42);

    println!("\nUMAP Parameters:");
    println!("  n_neighbors: 15");
    println!("  n_components: 2");
    println!("  min_dist: 0.1");
    println!("  learning_rate: 1.0");
    println!("  n_epochs: 200");
    println!("  random_state: 42");

    println!("\nRunning UMAP...");
    let embedding = umap.fit_transform(&data);

    println!("\nOutput embedding shape: {:?}", embedding.shape());

    // Calculate statistics
    let mut x_vals: Vec<f64> = embedding.column(0).iter().copied().collect();
    let mut y_vals: Vec<f64> = embedding.column(1).iter().copied().collect();

    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let x_mean: f64 = embedding.column(0).mean().unwrap();
    let y_mean: f64 = embedding.column(1).mean().unwrap();

    let x_std: f64 = (embedding.column(0).iter()
        .map(|&x| (x - x_mean).powi(2))
        .sum::<f64>() / (n_samples as f64 - 1.0)).sqrt();

    let y_std: f64 = (embedding.column(1).iter()
        .map(|&y| (y - y_mean).powi(2))
        .sum::<f64>() / (n_samples as f64 - 1.0)).sqrt();

    println!("\n=== Statistics ===");
    println!("Embedding range X: [{:.6}, {:.6}]", x_vals[0], x_vals[x_vals.len() - 1]);
    println!("Embedding range Y: [{:.6}, {:.6}]", y_vals[0], y_vals[y_vals.len() - 1]);
    println!("Embedding mean X: {:.6}", x_mean);
    println!("Embedding mean Y: {:.6}", y_mean);
    println!("Embedding sd X: {:.6}", x_std);
    println!("Embedding sd Y: {:.6}", y_std);

    // Save to CSV
    let mut file = File::create("results/rust_crabs_embedding.csv").expect("Failed to create file");
    writeln!(file, "V1,V2").expect("Failed to write header");
    for i in 0..embedding.nrows() {
        writeln!(file, "{},{}", embedding[[i, 0]], embedding[[i, 1]])
            .expect("Failed to write row");
    }

    println!("\nFile saved: results/rust_crabs_embedding.csv");
}
