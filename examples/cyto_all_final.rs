use ndarray::Array2;
use umaprs::{UMAP, KnnMethod, compute_knn_quant4_kdtree, compute_knn_quant8_kdtree};
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

fn run(data: &Array2<f64>, name: &str, path: &str, umap: &UMAP, timings: &mut Vec<(String, f64)>) {
    let t = Instant::now();
    let emb = umap.fit_transform(data);
    let elapsed = t.elapsed().as_secs_f64();
    println!("{:<18} {:.3}s", name, elapsed);
    save(&emb, path);
    timings.push((name.to_string(), elapsed));
}

fn run_knn(data: &Array2<f64>, name: &str, path: &str, knn: &Array2<usize>, umap: &UMAP, timings: &mut Vec<(String, f64)>) {
    let t = Instant::now();
    let emb = umap.fit_transform_with_knn(data, knn);
    let elapsed = t.elapsed().as_secs_f64();
    println!("{:<18} {:.3}s", name, elapsed);
    save(&emb, path);
    timings.push((name.to_string(), elapsed));
}

fn main() {
    let data = read_csv("data/cyto_data_clean.csv");
    println!("CyTOF: {} x {}\n", data.nrows(), data.ncols());

    let umap = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42);
    let mut timings: Vec<(String, f64)> = Vec::new();

    run(&data, "kd-tree", "results/cyto_emb_kdtree.csv", &umap, &mut timings);

    let t = Instant::now();
    let knn = compute_knn_quant4_kdtree(&data, 15);
    let emb = umap.fit_transform_with_knn(&data, &knn);
    let elapsed = t.elapsed().as_secs_f64();
    println!("{:<18} {:.3}s", "TQ4+kd+QJL", elapsed);
    save(&emb, "results/cyto_emb_tq4.csv");
    timings.push(("TQ4+kd+QJL".into(), elapsed));

    let t = Instant::now();
    let knn = compute_knn_quant8_kdtree(&data, 15);
    let emb = umap.fit_transform_with_knn(&data, &knn);
    let elapsed = t.elapsed().as_secs_f64();
    println!("{:<18} {:.3}s", "TQ8+kd+QJL", elapsed);
    save(&emb, "results/cyto_emb_tq8.csv");
    timings.push(("TQ8+kd+QJL".into(), elapsed));

    run(&data, "train 10%",
        "results/cyto_emb_train10.csv",
        &UMAP::new().n_neighbors(15).n_epochs(200).random_state(42).train_size(0.1),
        &mut timings);

    #[cfg(feature = "cuda")]
    {
        run(&data, "GPU f32", "results/cyto_emb_gpu.csv",
            &UMAP::new().n_neighbors(15).n_epochs(200).random_state(42).knn_method(KnnMethod::Gpu),
            &mut timings);

        run(&data, "GPU TQ4", "results/cyto_emb_gpu_tq4.csv",
            &UMAP::new().n_neighbors(15).n_epochs(200).random_state(42).knn_method(KnnMethod::GpuTQ4),
            &mut timings);
    }

    // Write timings CSV for R
    let mut f = File::create("results/timings.csv").unwrap();
    writeln!(f, "method,time").unwrap();
    for (name, time) in &timings {
        writeln!(f, "{},{:.3}", name, time).unwrap();
    }
}
