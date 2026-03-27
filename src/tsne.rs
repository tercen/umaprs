/// t-SNE (t-distributed Stochastic Neighbor Embedding)
///
/// Uses the same kNN infrastructure as UMAP:
/// - kd-tree, HNSW, or TQ compressed for neighbor search
/// - Sparse P matrix from kNN (not full n×n)
/// - Gradient descent with early exaggeration
///
/// The key difference from UMAP: Gaussian affinities in high-d,
/// Student-t (Cauchy) kernel in low-d, KL divergence loss.

use ndarray::Array2;
use rayon::prelude::*;
use crate::quadtree::QuadTree;

/// Compute pairwise conditional probabilities from kNN distances.
/// Uses binary search for per-point sigma to match target perplexity.
/// Returns sparse P matrix as (rows, cols, vals) triplets (symmetrized).
fn compute_p_matrix(
    knn_indices: &Array2<usize>,
    knn_dists: &Array2<f64>,
    perplexity: f64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = knn_indices.nrows();
    let k = knn_indices.ncols();
    let target_entropy = perplexity.ln();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        // Binary search for sigma_i such that perplexity matches
        let mut lo = 1e-10f64;
        let mut hi = 1e4f64;
        let mut sigma = 1.0;

        for _ in 0..64 {
            sigma = (lo + hi) / 2.0;
            let beta = 1.0 / (2.0 * sigma * sigma);

            // Compute conditional probabilities p(j|i)
            let mut sum_exp = 0.0;
            for ki in 0..k {
                sum_exp += (-beta * knn_dists[[i, ki]] * knn_dists[[i, ki]]).exp();
            }

            if sum_exp < 1e-20 { lo = sigma; continue; }

            // Entropy = -Σ p log p
            let mut entropy = 0.0;
            for ki in 0..k {
                let p = (-beta * knn_dists[[i, ki]] * knn_dists[[i, ki]]).exp() / sum_exp;
                if p > 1e-20 {
                    entropy -= p * p.ln();
                }
            }

            if (entropy - target_entropy).abs() < 1e-5 { break; }
            if entropy > target_entropy { hi = sigma; } else { lo = sigma; }
        }

        // Compute final p(j|i) with converged sigma
        let beta = 1.0 / (2.0 * sigma * sigma);
        let mut sum_exp = 0.0;
        for ki in 0..k {
            sum_exp += (-beta * knn_dists[[i, ki]] * knn_dists[[i, ki]]).exp();
        }

        for ki in 0..k {
            let j = knn_indices[[i, ki]];
            let p = (-beta * knn_dists[[i, ki]] * knn_dists[[i, ki]]).exp() / sum_exp.max(1e-20);
            if p > 1e-12 {
                rows.push(i);
                cols.push(j);
                vals.push(p);
            }
        }
    }

    // Symmetrize: P_ij = (p(j|i) + p(i|j)) / (2n)
    let mut sym: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
    for idx in 0..rows.len() {
        let i = rows[idx];
        let j = cols[idx];
        let v = vals[idx];
        *sym.entry((i, j)).or_insert(0.0) += v;
        *sym.entry((j, i)).or_insert(0.0) += v;
    }

    let scale = 1.0 / (2.0 * n as f64);
    let mut s_rows = Vec::new();
    let mut s_cols = Vec::new();
    let mut s_vals = Vec::new();
    for (&(i, j), &v) in &sym {
        let p = (v * scale).max(1e-12);
        s_rows.push(i);
        s_cols.push(j);
        s_vals.push(p);
    }

    (s_rows, s_cols, s_vals)
}

/// Compute kNN distances from data using kd-tree/HNSW
fn compute_knn_dists(
    data: &Array2<f64>,
    knn_indices: &Array2<usize>,
) -> Array2<f64> {
    let n = data.nrows();
    let k = knn_indices.ncols();
    let mut knn_dists = Array2::zeros((n, k));

    for i in 0..n {
        for ki in 0..k {
            let j = knn_indices[[i, ki]];
            let d: f64 = data.row(i).iter().zip(data.row(j).iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>().sqrt();
            knn_dists[[i, ki]] = d;
        }
    }

    knn_dists
}

/// Compact P matrix: CSR format with f32 values and u32 column indices.
/// At 1M points: ~600 MB vs ~3.6 GB for COO triplets.
pub struct CompactP {
    /// Row offsets: length n+1
    pub row_offsets: Vec<u32>,
    /// Column indices
    pub col_indices: Vec<u32>,
    /// Symmetrized P values (f32 for memory)
    pub values: Vec<f32>,
}

impl CompactP {
    pub fn from_triplets(n: usize, rows: &[usize], cols: &[usize], vals: &[f64]) -> Self {
        // Count entries per row
        let mut counts = vec![0u32; n];
        for &r in rows { counts[r] += 1; }

        let mut row_offsets = vec![0u32; n + 1];
        for i in 0..n { row_offsets[i + 1] = row_offsets[i] + counts[i]; }
        let nnz = row_offsets[n] as usize;

        let mut col_indices = vec![0u32; nnz];
        let mut values = vec![0.0f32; nnz];
        let mut pos = row_offsets[..n].to_vec(); // current insertion position per row

        for idx in 0..rows.len() {
            let r = rows[idx];
            let p = pos[r] as usize;
            col_indices[p] = cols[idx] as u32;
            values[p] = vals[idx] as f32;
            pos[r] += 1;
        }

        Self { row_offsets, col_indices, values }
    }

    pub fn memory_bytes(&self) -> usize {
        self.row_offsets.len() * 4 + self.col_indices.len() * 4 + self.values.len() * 4
    }
}

/// t-SNE with Barnes-Hut O(n log n) approximation, compact P matrix
pub fn tsne_optimize_bh_compact(
    embedding: &mut Array2<f64>,
    p: &CompactP,
    n_iter: usize,
    learning_rate: f64,
    early_exaggeration: f64,
    early_exaggeration_iter: usize,
    theta: f64,
) {
    let n = embedding.nrows();
    let mut gains_x = vec![1.0f64; n];
    let mut gains_y = vec![1.0f64; n];
    let mut vel_x = vec![0.0f64; n];
    let mut vel_y = vec![0.0f64; n];

    let mut ex = vec![0.0f64; n];
    let mut ey = vec![0.0f64; n];

    for iter in 0..n_iter {
        let momentum = if iter < 250 { 0.5 } else { 0.8 };
        let exag = if iter < early_exaggeration_iter { early_exaggeration as f32 } else { 1.0f32 };

        for i in 0..n { ex[i] = embedding[[i, 0]]; ey[i] = embedding[[i, 1]]; }

        let tree = QuadTree::build(&ex, &ey);
        let (rep_fx, rep_fy, z_sum) = tree.compute_repulsion(&ex, &ey, theta);

        let z_inv = 1.0 / z_sum.max(1e-20);
        let mut grad_x = vec![0.0f64; n];
        let mut grad_y = vec![0.0f64; n];

        for i in 0..n {
            grad_x[i] = -4.0 * z_inv * rep_fx[i];
            grad_y[i] = -4.0 * z_inv * rep_fy[i];
        }

        // Attractive from compact CSR
        for i in 0..n {
            let start = p.row_offsets[i] as usize;
            let end = p.row_offsets[i + 1] as usize;
            let xi = ex[i];
            let yi = ey[i];
            for idx in start..end {
                let j = p.col_indices[idx] as usize;
                let pv = p.values[idx] as f64 * exag as f64;
                let dx = xi - ex[j];
                let dy = yi - ey[j];
                let q = 1.0 / (1.0 + dx * dx + dy * dy);
                let mult = 4.0 * pv * q;
                grad_x[i] += mult * dx;
                grad_y[i] += mult * dy;
            }
        }

        for i in 0..n {
            let gx = grad_x[i];
            let gy = grad_y[i];
            if (gx > 0.0) != (vel_x[i] > 0.0) { gains_x[i] = (gains_x[i] + 0.2).min(10.0); }
            else { gains_x[i] = (gains_x[i] * 0.8).max(0.01); }
            if (gy > 0.0) != (vel_y[i] > 0.0) { gains_y[i] = (gains_y[i] + 0.2).min(10.0); }
            else { gains_y[i] = (gains_y[i] * 0.8).max(0.01); }
            vel_x[i] = momentum * vel_x[i] - learning_rate * gains_x[i] * gx;
            vel_y[i] = momentum * vel_y[i] - learning_rate * gains_y[i] * gy;
            embedding[[i, 0]] += vel_x[i];
            embedding[[i, 1]] += vel_y[i];
        }

        let mean_x = embedding.column(0).sum() / n as f64;
        let mean_y = embedding.column(1).sum() / n as f64;
        for i in 0..n {
            embedding[[i, 0]] -= mean_x;
            embedding[[i, 1]] -= mean_y;
        }
    }
}

/// t-SNE with Barnes-Hut O(n log n) approximation
pub fn tsne_optimize_bh(
    embedding: &mut Array2<f64>,
    p_rows: &[usize],
    p_cols: &[usize],
    p_vals: &[f64],
    n_iter: usize,
    learning_rate: f64,
    early_exaggeration: f64,
    early_exaggeration_iter: usize,
    theta: f64,
) {
    let n = embedding.nrows();
    let mut gains_x = vec![1.0f64; n];
    let mut gains_y = vec![1.0f64; n];
    let mut vel_x = vec![0.0f64; n];
    let mut vel_y = vec![0.0f64; n];

    // Pre-sort edges by source for cache-friendly access
    let mut sorted_edges: Vec<(usize, usize, f64)> = (0..p_rows.len())
        .map(|idx| (p_rows[idx], p_cols[idx], p_vals[idx]))
        .collect();
    sorted_edges.sort_unstable_by_key(|&(i, _, _)| i);
    let se_i: Vec<usize> = sorted_edges.iter().map(|e| e.0).collect();
    let se_j: Vec<usize> = sorted_edges.iter().map(|e| e.1).collect();
    let se_p: Vec<f64> = sorted_edges.iter().map(|e| e.2).collect();
    let n_edges = se_i.len();

    let mut ex = vec![0.0f64; n];
    let mut ey = vec![0.0f64; n];

    for iter in 0..n_iter {
        let momentum = if iter < 250 { 0.5 } else { 0.8 };
        let exag = if iter < early_exaggeration_iter { early_exaggeration } else { 1.0 };

        // Extract current positions (reuse buffers)
        for i in 0..n { ex[i] = embedding[[i, 0]]; ey[i] = embedding[[i, 1]]; }

        // Build quadtree and compute repulsive forces O(n log n)
        let t0 = std::time::Instant::now();
        let tree = QuadTree::build(&ex, &ey);
        let (rep_fx, rep_fy, z_sum) = tree.compute_repulsion(&ex, &ey, theta);

        let t0 = std::time::Instant::now();

        // Attractive + repulsive forces combined
        let z_inv = 1.0 / z_sum.max(1e-20);
        let mut grad_x = vec![0.0f64; n];
        let mut grad_y = vec![0.0f64; n];

        // Repulsive (from BH)
        for i in 0..n {
            grad_x[i] = -4.0 * z_inv * rep_fx[i];
            grad_y[i] = -4.0 * z_inv * rep_fy[i];
        }

        // Attractive (sorted by source for cache locality)
        for idx in 0..n_edges {
            let i = unsafe { *se_i.get_unchecked(idx) };
            let j = unsafe { *se_j.get_unchecked(idx) };
            let p = unsafe { *se_p.get_unchecked(idx) } * exag;
            let dx = unsafe { *ex.get_unchecked(i) - *ex.get_unchecked(j) };
            let dy = unsafe { *ey.get_unchecked(i) - *ey.get_unchecked(j) };
            let q = 1.0 / (1.0 + dx * dx + dy * dy);
            let mult = 4.0 * p * q;
            unsafe {
                *grad_x.get_unchecked_mut(i) += mult * dx;
                *grad_y.get_unchecked_mut(i) += mult * dy;
            }
        }

        let t0 = std::time::Instant::now();

        // Update with momentum + adaptive gains (flat arrays, no ndarray overhead)
        for i in 0..n {
            let gx = grad_x[i];
            let gy = grad_y[i];

            if (gx > 0.0) != (vel_x[i] > 0.0) { gains_x[i] = (gains_x[i] + 0.2).min(10.0); }
            else { gains_x[i] = (gains_x[i] * 0.8).max(0.01); }
            if (gy > 0.0) != (vel_y[i] > 0.0) { gains_y[i] = (gains_y[i] + 0.2).min(10.0); }
            else { gains_y[i] = (gains_y[i] * 0.8).max(0.01); }

            vel_x[i] = momentum * vel_x[i] - learning_rate * gains_x[i] * gx;
            vel_y[i] = momentum * vel_y[i] - learning_rate * gains_y[i] * gy;
            embedding[[i, 0]] += vel_x[i];
            embedding[[i, 1]] += vel_y[i];
        }

        // Center
        let mean_x = embedding.column(0).sum() / n as f64;
        let mean_y = embedding.column(1).sum() / n as f64;
        for i in 0..n {
            embedding[[i, 0]] -= mean_x;
            embedding[[i, 1]] -= mean_y;
        }
    }
}

/// t-SNE exact gradient descent (O(n²) — for reference/small datasets)
pub fn tsne_optimize(
    embedding: &mut Array2<f64>,
    p_rows: &[usize],
    p_cols: &[usize],
    p_vals: &[f64],
    n_iter: usize,
    learning_rate: f64,
    early_exaggeration: f64,
    early_exaggeration_iter: usize,
) {
    let n = embedding.nrows();
    let mut gains = Array2::from_elem((n, 2), 1.0f64);
    let mut velocity = Array2::zeros((n, 2));
    let momentum_init = 0.5;
    let momentum_final = 0.8;

    for iter in 0..n_iter {
        let momentum = if iter < 250 { momentum_init } else { momentum_final };
        let exag = if iter < early_exaggeration_iter { early_exaggeration } else { 1.0 };

        // Compute Q matrix denominator (Student-t kernel)
        // Q_ij = (1 + ||y_i - y_j||²)^{-1} / Σ_{k≠l} (1 + ||y_k - y_l||²)^{-1}
        let mut q_denom = 0.0f64;
        // For O(n²) exact computation:
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = embedding[[i, 0]] - embedding[[j, 0]];
                let dy = embedding[[i, 1]] - embedding[[j, 1]];
                let d2 = dx * dx + dy * dy;
                q_denom += 2.0 / (1.0 + d2); // ×2 for symmetry
            }
        }
        q_denom = q_denom.max(1e-20);

        // Compute gradients
        let mut grad = Array2::zeros((n, 2));

        // Attractive forces (sparse, from P matrix)
        for idx in 0..p_rows.len() {
            let i = p_rows[idx];
            let j = p_cols[idx];
            let p = p_vals[idx] * exag;

            let dx = embedding[[i, 0]] - embedding[[j, 0]];
            let dy = embedding[[i, 1]] - embedding[[j, 1]];
            let d2 = dx * dx + dy * dy;
            let q_unnorm = 1.0 / (1.0 + d2);

            let mult = 4.0 * (p - q_unnorm / q_denom) * q_unnorm;
            grad[[i, 0]] += mult * dx;
            grad[[i, 1]] += mult * dy;
        }

        // Repulsive forces (O(n²) — exact)
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = embedding[[i, 0]] - embedding[[j, 0]];
                let dy = embedding[[i, 1]] - embedding[[j, 1]];
                let d2 = dx * dx + dy * dy;
                let q_unnorm = 1.0 / (1.0 + d2);
                let q = q_unnorm / q_denom;

                let mult = -4.0 * q * q_unnorm;
                grad[[i, 0]] += mult * dx;
                grad[[i, 1]] += mult * dy;
                grad[[j, 0]] -= mult * dx;
                grad[[j, 1]] -= mult * dy;
            }
        }

        // Update with momentum and adaptive gains
        for i in 0..n {
            for c in 0..2usize {
                let g: f64 = grad[[i, c]];
                let v: f64 = velocity[[i, c]];

                if (g > 0.0) != (v > 0.0) {
                    gains[[i, c]] = (gains[[i, c]] + 0.2).min(10.0);
                } else {
                    gains[[i, c]] = (gains[[i, c]] * 0.8).max(0.01);
                }

                velocity[[i, c]] = momentum * v - learning_rate * gains[[i, c]] * g;
                embedding[[i, c]] += velocity[[i, c]];
            }
        }

        // Center embedding
        let mean_x = embedding.column(0).sum() / n as f64;
        let mean_y = embedding.column(1).sum() / n as f64;
        for i in 0..n {
            embedding[[i, 0]] -= mean_x;
            embedding[[i, 1]] -= mean_y;
        }
    }
}

/// Run t-SNE on data with given kNN indices
pub fn run_tsne(
    data: &Array2<f64>,
    knn_indices: &Array2<usize>,
    perplexity: f64,
    n_iter: usize,
    learning_rate: f64,
    random_state: Option<u64>,
) -> Array2<f64> {
    let n = data.nrows();

    eprintln!("t-SNE: {} points, perplexity={}", n, perplexity);

    // Compute kNN distances
    let knn_dists = compute_knn_dists(data, knn_indices);

    // Compute P matrix
    eprintln!("  Computing P matrix...");
    let (p_rows, p_cols, p_vals) = compute_p_matrix(knn_indices, &knn_dists, perplexity);
    eprintln!("  {} non-zero P entries", p_rows.len());

    // Initialize with PCA
    eprintln!("  PCA initialization...");
    let mut embedding = crate::spectral::pca_initialization(data, 2, random_state);

    // Scale down for t-SNE (typical initial scale ~0.01)
    embedding.mapv_inplace(|x| x * 0.01);

    // Compact P matrix
    let compact = CompactP::from_triplets(n, &p_rows, &p_cols, &p_vals);
    eprintln!("  P matrix: {} MB (compact CSR)", compact.memory_bytes() / 1024 / 1024);
    drop(p_rows); drop(p_cols); drop(p_vals);

    // Optimize with Barnes-Hut
    eprintln!("  Optimizing ({} iterations, Barnes-Hut theta=0.5)...", n_iter);
    tsne_optimize_bh_compact(
        &mut embedding,
        &compact,
        n_iter,
        learning_rate,
        12.0,
        250,
        0.5,
    );

    embedding
}

/// Run t-SNE on compressed data — memory efficient.
/// Computes kNN distances on-the-fly from compressed representation,
/// never materializes the full kNN distance matrix.
pub fn run_tsne_compressed(
    qdata: &crate::quantize::QuantizedData,
    perplexity: f64,
    n_iter: usize,
    learning_rate: f64,
    random_state: Option<u64>,
) -> Array2<f64> {
    let n = qdata.n_samples;
    let k = (3.0 * perplexity) as usize + 1;

    eprintln!("t-SNE compressed: {} points, perplexity={}", n, perplexity);

    // kNN from compressed distances
    eprintln!("  kNN from compressed distances...");
    let knn = crate::compressed::knn_compressed(qdata, k);

    // Compute P matrix directly from compressed distances — no kNN distance matrix
    eprintln!("  Computing P matrix (on-the-fly distances)...");
    let (p_rows, p_cols, p_vals) = compute_p_matrix_compressed(&knn, qdata, perplexity);
    eprintln!("  {} non-zero P entries", p_rows.len());

    // Compact P + drop intermediates
    let compact = CompactP::from_triplets(n, &p_rows, &p_cols, &p_vals);
    eprintln!("  P matrix: {} MB (compact CSR)", compact.memory_bytes() / 1024 / 1024);
    drop(p_rows); drop(p_cols); drop(p_vals);
    drop(knn);

    // PCA init
    eprintln!("  PCA initialization...");
    let mut embedding = crate::compressed::pca_compressed(qdata, 2, random_state);
    embedding.mapv_inplace(|x| x * 0.01);

    // Optimize
    eprintln!("  Optimizing ({} iterations, Barnes-Hut theta=0.5)...", n_iter);
    tsne_optimize_bh_compact(
        &mut embedding,
        &compact,
        n_iter,
        learning_rate,
        12.0,
        250,
        0.5,
    );

    embedding
}

/// Compute P matrix directly from compressed data — no intermediate distance matrix.
/// For each point, binary-search sigma using on-the-fly TQ distances to kNN neighbors.
fn compute_p_matrix_compressed(
    knn_indices: &Array2<usize>,
    qdata: &crate::quantize::QuantizedData,
    perplexity: f64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = knn_indices.nrows();
    let k = knn_indices.ncols();
    let target_entropy = perplexity.ln();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        // Compute distances on-the-fly from compressed data
        let mut dists = Vec::with_capacity(k);
        for ki in 0..k {
            let j = knn_indices[[i, ki]];
            dists.push((qdata.approx_dist_sq(i, j) as f64).sqrt());
        }

        // Binary search for sigma
        let mut lo = 1e-10f64;
        let mut hi = 1e4f64;
        let mut sigma = 1.0;

        for _ in 0..64 {
            sigma = (lo + hi) / 2.0;
            let beta = 1.0 / (2.0 * sigma * sigma);

            let mut sum_exp = 0.0;
            for &d in &dists {
                sum_exp += (-beta * d * d).exp();
            }
            if sum_exp < 1e-20 { lo = sigma; continue; }

            let mut entropy = 0.0;
            for &d in &dists {
                let p = (-beta * d * d).exp() / sum_exp;
                if p > 1e-20 { entropy -= p * p.ln(); }
            }

            if (entropy - target_entropy).abs() < 1e-5 { break; }
            if entropy > target_entropy { hi = sigma; } else { lo = sigma; }
        }

        // Final probabilities
        let beta = 1.0 / (2.0 * sigma * sigma);
        let mut sum_exp = 0.0;
        for &d in &dists {
            sum_exp += (-beta * d * d).exp();
        }

        for ki in 0..k {
            let j = knn_indices[[i, ki]];
            let p = (-beta * dists[ki] * dists[ki]).exp() / sum_exp.max(1e-20);
            if p > 1e-12 {
                rows.push(i);
                cols.push(j);
                vals.push(p);
            }
        }
    }

    // Symmetrize
    let mut sym: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
    for idx in 0..rows.len() {
        *sym.entry((rows[idx], cols[idx])).or_insert(0.0) += vals[idx];
        *sym.entry((cols[idx], rows[idx])).or_insert(0.0) += vals[idx];
    }

    let scale = 1.0 / (2.0 * n as f64);
    let mut s_rows = Vec::new();
    let mut s_cols = Vec::new();
    let mut s_vals = Vec::new();
    for (&(i, j), &v) in &sym {
        s_vals.push((v * scale).max(1e-12));
        s_rows.push(i);
        s_cols.push(j);
    }

    (s_rows, s_cols, s_vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsne_basic() {
        let data = Array2::from_shape_vec((20, 4), (0..80).map(|x| x as f64).collect()).unwrap();
        let knn = Array2::from_shape_fn((20, 3), |(i, k)| (i + k + 1) % 20);
        let emb = run_tsne(&data, &knn, 5.0, 50, 200.0, Some(42));
        assert_eq!(emb.shape(), &[20, 2]);
    }
}
