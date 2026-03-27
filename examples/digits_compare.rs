use ndarray::Array2;
use umaprs::{UMAP, compute_knn_bruteforce, compute_knn_quant_hnsw, compute_knn_quant_hnsw_8bit};
use std::fs::File;
use std::io::{Write, BufReader, BufRead};
use std::time::Instant;

fn read_csv(path: &str) -> Array2<f64> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    lines.next(); // skip header

    let mut data_vec = Vec::new();
    let mut n_rows = 0;
    for line in lines {
        let line = line.unwrap();
        let values: Vec<f64> = line.split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();
        data_vec.extend(values);
        n_rows += 1;
    }
    let n_cols = data_vec.len() / n_rows;
    Array2::from_shape_vec((n_rows, n_cols), data_vec).unwrap()
}

fn save_embedding(embedding: &Array2<f64>, path: &str) {
    let mut f = File::create(path).unwrap();
    writeln!(f, "V1,V2").unwrap();
    for i in 0..embedding.nrows() {
        writeln!(f, "{},{}", embedding[[i, 0]], embedding[[i, 1]]).unwrap();
    }
}

fn main() {
    let data = read_csv("data/digits_data.csv");
    println!("Optical Digits: {} samples, {} features\n", data.nrows(), data.ncols());

    let umap = UMAP::new()
        .n_neighbors(15)
        .n_components(2)
        .min_dist(0.1)
        .learning_rate(1.0)
        .n_epochs(200)
        .random_state(42);

    // 1. TurboQuant 4-bit + HNSW (default for this size+dim)
    eprintln!("=== TurboQuant 4-bit + HNSW ===");
    let start = Instant::now();
    let emb_tq4 = umap.fit_transform(&data); // auto-selects TQ+HNSW
    let t_tq4 = start.elapsed();
    save_embedding(&emb_tq4, "results/digits_emb_tq4.csv");
    println!("TQ 4-bit:  {:.2}s", t_tq4.as_secs_f64());

    // 2. TurboQuant 8-bit + HNSW
    eprintln!("\n=== TurboQuant 8-bit + HNSW ===");
    let start = Instant::now();
    let knn_tq8 = compute_knn_quant_hnsw_8bit(&data, 15);
    let emb_tq8 = umap.fit_transform_with_knn(&data, &knn_tq8);
    let t_tq8 = start.elapsed();
    save_embedding(&emb_tq8, "results/digits_emb_tq8.csv");
    println!("TQ 8-bit:  {:.2}s", t_tq8.as_secs_f64());

    println!("\nSaved digits_emb_*.csv");
}
