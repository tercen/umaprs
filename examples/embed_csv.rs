//! Embed a CSV at three seeds for scripts/envelope.py:
//!
//!     cargo run --release --example embed_csv -- <name> <data.csv> [min_dist] [knn=auto|brute|kdtree|hnsw]
//!
//! Writes results/env_<name>_umaprs_seed{1,2,3}.csv and a timings file. Same settings as the
//! umap-learn and uwot runs the scorer compares against: n_neighbors 15, n_epochs 200.
use ndarray::Array2;
use std::io::Write;
use std::time::Instant;

fn read_csv(path: &str) -> Array2<f64> {
    let text = std::fs::read_to_string(path).expect("read data csv");
    let rows: Vec<Vec<f64>> = text
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|v| v.trim().parse().expect("numeric cell"))
                .collect()
        })
        .collect();
    let (n, d) = (rows.len(), rows[0].len());
    Array2::from_shape_vec((n, d), rows.into_iter().flatten().collect()).unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (name, path) = (&a[1], &a[2]);
    let md: f64 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.01);
    let knn = a
        .get(4)
        .map(|v| v.trim_start_matches("knn=").to_string())
        .unwrap_or("auto".into());
    let (method, engine) = match knn.as_str() {
        "brute" => (umaprs::KnnMethod::BruteForce, "umaprs-brute"),
        "kdtree" => (umaprs::KnnMethod::KdTree, "umaprs-kdtree"),
        "hnsw" => (umaprs::KnnMethod::Hnsw, "umaprs-hnsw"),
        _ => (umaprs::KnnMethod::Auto, "umaprs"),
    };
    let data = read_csv(path);
    std::fs::create_dir_all("results").unwrap();
    let mut secs = Vec::new();
    for seed in 1..=3u64 {
        let t = Instant::now();
        let emb = umaprs::UMAP::new()
            .n_neighbors(15)
            .min_dist(md)
            .n_epochs(200)
            .random_state(seed)
            .knn_method(method.clone())
            .fit_transform(&data);
        secs.push(t.elapsed().as_secs_f64());
        let mut f =
            std::fs::File::create(format!("results/env_{name}_{engine}_seed{seed}.csv")).unwrap();
        for i in 0..emb.nrows() {
            writeln!(f, "{:.8e},{:.8e}", emb[[i, 0]], emb[[i, 1]]).unwrap();
        }
    }
    std::fs::write(
        format!("results/env_{name}_{engine}_timings.json"),
        format!(
            "[{}]",
            secs.iter()
                .map(|s| format!("{s:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        ),
    )
    .unwrap();
    eprintln!("{engine} {name}: {:?} s", secs);
}
