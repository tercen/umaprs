use ndarray::Array2;
use umaprs::UMAP;
use std::fs::File;
use std::io::{Write, BufReader, BufRead};
use std::time::Instant;

fn read_csv(path: &str) -> Array2<f64> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    lines.next();
    let mut v = Vec::new();
    let mut n = 0;
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
    println!("CyTOF: {} cells x {} markers\n", data.nrows(), data.ncols());

    // Exact same params as uwot defaults
    let umap = UMAP::new()
        .n_neighbors(15)
        .n_components(2)
        .min_dist(0.1)
        .spread(1.0)
        .learning_rate(1.0)
        .n_epochs(200)
        .negative_sample_rate(5.0)
        .repulsion_strength(1.0)
        .random_state(42);

    for run in 1..=3 {
        let start = Instant::now();
        let embedding = umap.fit_transform(&data);
        let t = start.elapsed();
        println!("Rust run {}: {:.3}s", run, t.as_secs_f64());

        if run == 1 {
            let mut f = File::create("results/cyto_emb_rust.csv").unwrap();
            writeln!(f, "V1,V2").unwrap();
            for i in 0..embedding.nrows() {
                writeln!(f, "{},{}", embedding[[i, 0]], embedding[[i, 1]]).unwrap();
            }
        }
    }
}
