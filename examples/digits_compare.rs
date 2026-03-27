use ndarray::Array2;
use umaprs::{UMAP, QuantBits};
use std::fs::File;
use std::io::{Write, BufReader, BufRead};
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

fn save(e: &Array2<f64>, p: &str) {
    let mut f = File::create(p).unwrap();
    writeln!(f, "V1,V2").unwrap();
    for i in 0..e.nrows() { writeln!(f, "{},{}", e[[i,0]], e[[i,1]]).unwrap(); }
}

fn main() {
    let data = read_csv("data/digits_data.csv");
    println!("Optical Digits: {} samples, {} features\n", data.nrows(), data.ncols());

    let umap = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42);

    // Standard
    let t = Instant::now();
    let emb = umap.fit_transform(&data);
    println!("Standard:       {:.2}s", t.elapsed().as_secs_f64());
    save(&emb, "results/digits_emb_standard.csv");

    // Compressed TQ8
    let t = Instant::now();
    let emb = umap.fit_transform_compressed(&data, QuantBits::Eight);
    println!("Compressed TQ8: {:.2}s", t.elapsed().as_secs_f64());
    save(&emb, "results/digits_emb_tq8.csv");
}
