use ndarray::Array2;
use umaprs::{UMAP, KnnMethod};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::time::Instant;

fn read_csv(path: &str) -> Array2<f64> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    lines.next();
    let mut v = Vec::new(); let mut n = 0;
    for line in lines { let l = line.unwrap(); v.extend(l.split(',').map(|s| s.trim().parse::<f64>().unwrap())); n += 1; }
    let c = v.len() / n;
    Array2::from_shape_vec((n, c), v).unwrap()
}

fn main() {
    // Tiny test first
    println!("=== Tiny test (8 points) ===");
    let tiny = Array2::from_shape_vec((8, 4), vec![
        0.0,0.0,0.0,0.0, 1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 1.0,1.0,0.0,0.0,
        10.0,10.0,10.0,10.0, 11.0,10.0,10.0,10.0, 10.0,11.0,10.0,10.0, 11.0,11.0,10.0,10.0,
    ]).unwrap();

    let emb = UMAP::new().n_neighbors(3).n_epochs(10).random_state(42)
        .knn_method(KnnMethod::Gpu).fit_transform(&tiny);
    println!("GPU f32 OK: {:?}", emb.shape());

    let emb = UMAP::new().n_neighbors(3).n_epochs(10).random_state(42)
        .knn_method(KnnMethod::GpuTQ4).fit_transform(&tiny);
    println!("GPU TQ4 OK: {:?}", emb.shape());

    // CyTOF benchmark
    println!("\n=== CyTOF 50k ===");
    let data = read_csv("data/cyto_data_clean.csv");
    println!("Shape: {} x {}\n", data.nrows(), data.ncols());

    let t = Instant::now();
    let _ = UMAP::new().n_epochs(200).random_state(42).fit_transform(&data);
    println!("CPU kd-tree:    {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let _ = UMAP::new().n_epochs(200).random_state(42)
        .knn_method(KnnMethod::Gpu).fit_transform(&data);
    println!("GPU f32:        {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let _ = UMAP::new().n_epochs(200).random_state(42)
        .knn_method(KnnMethod::GpuTQ4).fit_transform(&data);
    println!("GPU TQ4:        {:.3}s", t.elapsed().as_secs_f64());
}
