//! Recall of the HNSW kNN against brute force, across the beam width and the refine multiple.
//!
//!     cargo run --release --example knn_recall -- <data.csv>
//!
//! Prints recall@15 (fraction of the true 15 neighbours found) and the time, per setting.
use ndarray::Array2;
use std::time::Instant;
fn read_csv(path: &str) -> Array2<f64> {
    let text = std::fs::read_to_string(path).expect("read csv");
    let rows: Vec<Vec<f64>> = text
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .collect();
    let (n, d) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((n, d), rows.into_iter().flatten().collect()).unwrap()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = read_csv(&a[1]);
    let k = 15;
    let t = Instant::now();
    let exact = umaprs::compute_knn_bruteforce(&data, k);
    println!("brute force: {:.1} s", t.elapsed().as_secs_f64());
    let truth: Vec<std::collections::HashSet<usize>> = (0..exact.nrows())
        .map(|i| exact.row(i).iter().copied().collect())
        .collect();
    for &(ef, refine) in &[
        (30usize, 2usize),
        (60, 2),
        (100, 2),
        (100, 4),
        (200, 2),
        (200, 4),
        (400, 4),
    ] {
        let t = Instant::now();
        let got = umaprs::compute_knn_hnsw_tuned(&data, k, 42, ef, refine);
        let secs = t.elapsed().as_secs_f64();
        let hit: usize = (0..got.nrows())
            .map(|i| got.row(i).iter().filter(|j| truth[i].contains(j)).count())
            .sum();
        println!(
            "ef {ef:>3} refine {refine}x: recall@{k} {:.4}  {secs:.1} s",
            hit as f64 / (got.nrows() * k) as f64
        );
    }
}
