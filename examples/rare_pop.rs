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
    for &(pct, name) in &[("01pct", "0.1%"), ("1pct", "1%")] {
        let data = read_csv(&format!("data/aml_{}_data.csv", pct));
        println!("\n=== AML spike-in {} ({} cells, {} dims) ===\n", name, data.nrows(), data.ncols());

        let umap = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42);

        // 1. Standard (exact, full memory)
        let t = Instant::now();
        let emb = umap.fit_transform(&data);
        println!("Standard:            {:.3}s", t.elapsed().as_secs_f64());
        save(&emb, &format!("results/aml_{}_standard.csv", pct));

        // 2. Subsample 8% (same memory as TQ4)
        let t = Instant::now();
        let emb = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42)
            .train_size(0.08).fit_transform(&data);
        println!("Subsample 8%%:        {:.3}s", t.elapsed().as_secs_f64());
        save(&emb, &format!("results/aml_{}_sub8.csv", pct));

        // 3. Subsample 14% (same memory as TQ8)
        let t = Instant::now();
        let emb = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42)
            .train_size(0.14).fit_transform(&data);
        println!("Subsample 14%%:       {:.3}s", t.elapsed().as_secs_f64());
        save(&emb, &format!("results/aml_{}_sub14.csv", pct));

        // 4. TQ8 compressed
        let t = Instant::now();
        let emb = umap.fit_transform_compressed(&data, QuantBits::Eight);
        println!("TQ8 compressed:      {:.3}s", t.elapsed().as_secs_f64());
        save(&emb, &format!("results/aml_{}_tq8.csv", pct));
    }
}
