use ndarray::Array2;
use umaprs::{UMAP, KnnMethod};
use std::fs::File;
use std::io::{Write, BufReader, BufRead};
use std::time::Instant;

fn read_csv(path: &str) -> Array2<f64> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    lines.next();
    let mut v = Vec::new(); let mut n = 0;
    for line in lines {
        let l = line.unwrap();
        v.extend(l.split(',').map(|s| s.trim().parse::<f64>().unwrap()));
        n += 1;
    }
    let c = v.len() / n;
    Array2::from_shape_vec((n, c), v).unwrap()
}

fn main() {
    let data = read_csv("data/cyto_data_clean.csv");
    println!("CyTOF: {} x {}\n", data.nrows(), data.ncols());

    // CPU (kd-tree)
    let t = Instant::now();
    let emb_cpu = UMAP::new().n_epochs(200).random_state(42)
        .fit_transform(&data);
    println!("CPU (kd-tree): {:.3}s", t.elapsed().as_secs_f64());

    // GPU
    let t = Instant::now();
    let emb_gpu = UMAP::new().n_epochs(200).random_state(42)
        .knn_method(KnnMethod::Gpu)
        .fit_transform(&data);
    println!("GPU (cuBLAS):  {:.3}s", t.elapsed().as_secs_f64());

    // Save GPU result
    let mut f = File::create("results/cyto_emb_gpu.csv").unwrap();
    writeln!(f, "V1,V2").unwrap();
    for i in 0..emb_gpu.nrows() {
        writeln!(f, "{},{}", emb_gpu[[i, 0]], emb_gpu[[i, 1]]).unwrap();
    }
    println!("\nSaved: results/cyto_emb_gpu.csv");
}
