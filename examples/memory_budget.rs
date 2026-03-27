use ndarray::Array2;
use umaprs::{UMAP, QuantBits};
use std::fs::File;
use std::io::{Write, BufReader, BufRead};
use std::time::Instant;

fn read_csv(path: &str) -> Array2<f64> {
    let f = File::open(path).unwrap();
    let r = BufReader::new(f);
    let mut l = r.lines(); l.next();
    let mut v = Vec::new(); let mut n = 0;
    for line in l { let l = line.unwrap(); v.extend(l.split(',').map(|s| s.trim().parse::<f64>().unwrap())); n += 1; }
    Array2::from_shape_vec((n, v.len()/n), v).unwrap()
}

fn save(e: &Array2<f64>, p: &str) {
    let mut f = File::create(p).unwrap();
    writeln!(f, "V1,V2").unwrap();
    for i in 0..e.nrows() { writeln!(f, "{},{}", e[[i,0]], e[[i,1]]).unwrap(); }
}

fn main() {
    let data = read_csv("data/cyto_data_clean.csv");
    let n = data.nrows();
    let d = data.ncols();
    let full_mem = (n * d * 8) as f64 / 1024.0 / 1024.0;
    println!("CyTOF: {} x {}, {:.1} MB as f64\n", n, d, full_mem);

    let umap = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42);

    // === Memory budget ~1.7 MB (TQ8 equivalent) ===
    let tq8_ratio = 7.0; // approx compression ratio
    let train_frac_8 = 1.0 / tq8_ratio;
    println!("=== Budget ~1.7 MB (TQ8 = 7x compression) ===");
    println!("  TQ8 compressed: all {} points at 8-bit", n);
    println!("  Standard subsample: {:.0} points at f64 (train_size={:.3})\n",
             n as f64 * train_frac_8, train_frac_8);

    let t = Instant::now();
    let emb = umap.fit_transform_compressed(&data, QuantBits::Eight);
    println!("  TQ8 compressed:     {:.3}s", t.elapsed().as_secs_f64());
    save(&emb, "results/budget_tq8.csv");

    let t = Instant::now();
    let emb = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42)
        .train_size(train_frac_8).fit_transform(&data);
    println!("  Std train {:.0}%:      {:.3}s", train_frac_8 * 100.0, t.elapsed().as_secs_f64());
    save(&emb, "results/budget_std_7x.csv");

    // === Memory budget ~1.0 MB (TQ4 equivalent) ===
    let tq4_ratio = 12.0;
    let train_frac_4 = 1.0 / tq4_ratio;
    println!("\n=== Budget ~1.0 MB (TQ4 = 12x compression) ===");
    println!("  TQ4 compressed: all {} points at 4-bit", n);
    println!("  Standard subsample: {:.0} points at f64 (train_size={:.3})\n",
             n as f64 * train_frac_4, train_frac_4);

    let t = Instant::now();
    let emb = umap.fit_transform_compressed(&data, QuantBits::Four);
    println!("  TQ4 compressed:     {:.3}s", t.elapsed().as_secs_f64());
    save(&emb, "results/budget_tq4.csv");

    let t = Instant::now();
    let emb = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42)
        .train_size(train_frac_4).fit_transform(&data);
    println!("  Std train {:.0}%:       {:.3}s", train_frac_4 * 100.0, t.elapsed().as_secs_f64());
    save(&emb, "results/budget_std_12x.csv");
}
