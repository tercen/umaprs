use ndarray::Array2;
use umaprs::{UMAP, QuantBits, QuantizedData, compute_knn_graph};
use umaprs::tsne;
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
    // Use 10k subset for t-SNE (O(n²) repulsion)
    let n_sub = 10000;
    let data = data.slice(ndarray::s![..n_sub, ..]).to_owned();
    println!("CyTOF subset: {} x {}\n", data.nrows(), data.ncols());

    let perplexity = 30.0;
    let k = (3.0 * perplexity) as usize + 1;

    // Standard t-SNE
    let t = Instant::now();
    let knn = compute_knn_graph(&data, k);
    let emb = tsne::run_tsne(&data, &knn, perplexity, 1000, 200.0, Some(42));
    println!("t-SNE standard:     {:.3}s", t.elapsed().as_secs_f64());
    save(&emb, "results/tsne_standard.csv");

    // t-SNE compressed TQ8
    let t = Instant::now();
    let qdata = QuantizedData::encode_with_bits(&data, 42, QuantBits::Eight);
    let emb = tsne::run_tsne_compressed(&qdata, perplexity, 1000, 200.0, Some(42));
    println!("t-SNE TQ8:          {:.3}s", t.elapsed().as_secs_f64());
    save(&emb, "results/tsne_tq8.csv");

    // t-SNE compressed TQ4
    let t = Instant::now();
    let qdata4 = QuantizedData::encode_with_bits(&data, 42, QuantBits::Four);
    let emb = tsne::run_tsne_compressed(&qdata4, perplexity, 1000, 200.0, Some(42));
    println!("t-SNE TQ4:          {:.3}s", t.elapsed().as_secs_f64());
    save(&emb, "results/tsne_tq4.csv");

    // UMAP for comparison
    let t = Instant::now();
    let emb = UMAP::new().n_neighbors(15).n_epochs(200).random_state(42).fit_transform(&data);
    println!("UMAP standard:      {:.3}s", t.elapsed().as_secs_f64());
    save(&emb, "results/umap_10k.csv");
}
