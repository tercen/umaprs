use ndarray::Array2;
use rayon::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

use crate::kdtree::KdTree;

/// Sampling strategy for selecting training subset from full data.
///
/// Future options to consider:
///
/// - **Geometric sketching** (Hie et al., 2019, Cell Systems):
///   Hash points into hypercubes in feature space, pick one per occupied cell.
///   Gives uniform spatial coverage regardless of density — preserves rare
///   populations (e.g., rare stem cells among abundant T-cells).
///   Complexity: O(n) with hashing, but needs tuning of grid resolution.
///
/// - **Max-min diversity**:
///   Greedily pick the point farthest from all already-selected points.
///   Fills gaps in the selection, deterministic, no hyperparameters.
///   Complexity: O(n × m) where m = sample size — can be approximated
///   with kd-tree queries for nearest-selected-point.
#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    /// Random uniform sampling (default). Simple, fast, works well when
    /// populations are roughly balanced.
    Random,
}

/// A fitted UMAP model that can transform new data.
pub struct UmapModel {
    /// Training data (n_train × n_dims)
    pub training_data: Array2<f64>,
    /// Training embedding (n_train × n_components)
    pub embedding: Array2<f64>,
    /// Per-point sigma from smooth kNN distances
    pub sigmas: Vec<f64>,
    /// Per-point rho (distance to nearest neighbor)
    pub rhos: Vec<f64>,
    /// Curve parameter a
    pub a: f64,
    /// Curve parameter b
    pub b: f64,
    /// Number of neighbors
    pub n_neighbors: usize,
    /// Feature names (optional)
    pub feature_names: Option<Vec<String>>,
}

impl UmapModel {
    /// Sample a training subset from data.
    /// Returns (train_indices, train_data).
    pub fn sample_train(
        data: &Array2<f64>,
        train_fraction: f64,
        strategy: &SamplingStrategy,
        seed: u64,
    ) -> (Vec<usize>, Array2<f64>) {
        let n = data.nrows();
        let n_train = ((n as f64 * train_fraction).ceil() as usize).max(1).min(n);

        let indices = match strategy {
            SamplingStrategy::Random => {
                let mut idx: Vec<usize> = (0..n).collect();
                let mut rng = SmallRng::seed_from_u64(seed);
                idx.shuffle(&mut rng);
                idx.truncate(n_train);
                idx.sort(); // keep order stable for reproducibility
                idx
            }
        };

        let n_dims = data.ncols();
        let mut train = Array2::zeros((indices.len(), n_dims));
        for (i, &orig) in indices.iter().enumerate() {
            for j in 0..n_dims {
                train[[i, j]] = data[[orig, j]];
            }
        }

        (indices, train)
    }

    /// Transform new data points onto the existing embedding.
    pub fn transform(&self, new_data: &Array2<f64>) -> Array2<f64> {
        let expected_dims = self.training_data.ncols();
        if new_data.ncols() != expected_dims {
            let feat_info = match &self.feature_names {
                Some(names) => format!(": {}", names.join(", ")),
                None => String::new(),
            };
            panic!(
                "Input has {} features but model expects {}{}",
                new_data.ncols(), expected_dims, feat_info
            );
        }

        let n_new = new_data.nrows();
        let n_dims = new_data.ncols();
        let n_components = self.embedding.ncols();
        let n_train = self.training_data.nrows();
        let k = self.n_neighbors;

        // Build kd-tree on training data
        let flat: Vec<f32> = self.training_data.iter().map(|&v| v as f32).collect();
        let tree = KdTree::build(&flat, n_train, n_dims);

        let query_flat: Vec<f32> = new_data.iter().map(|&v| v as f32).collect();

        let a = self.a as f32;
        let b = self.b as f32;

        let results: Vec<Vec<f64>> = (0..n_new)
            .into_par_iter()
            .map(|qi| {
                let q_slice = &query_flat[qi * n_dims..(qi + 1) * n_dims];

                // Find k nearest neighbors in training data via brute scan
                // (kd-tree.knn needs internal index; for external queries use direct scan)
                let mut dists: Vec<(usize, f32)> = (0..n_train)
                    .map(|ti| {
                        let t_slice = &flat[ti * n_dims..(ti + 1) * n_dims];
                        let d: f32 = q_slice.iter().zip(t_slice.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        (ti, d)
                    })
                    .collect();

                dists.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
                dists.truncate(k);
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let nn_dists: Vec<f64> = dists.iter().map(|&(_, d)| (d as f64).sqrt()).collect();
                let nn_indices: Vec<usize> = dists.iter().map(|&(i, _)| i).collect();

                // Fuzzy weights
                let rho = nn_dists[0].max(0.0);
                let sigma: f64 = nn_indices.iter()
                    .map(|&i| self.sigmas[i])
                    .sum::<f64>() / k as f64;

                let mut weights = Vec::with_capacity(k);
                for &d in &nn_dists {
                    weights.push((-((d - rho).max(0.0) / sigma.max(1e-10))).exp());
                }

                // Initialize as weighted average of neighbor embeddings
                let weight_sum: f64 = weights.iter().sum();
                let mut pos = vec![0.0f64; n_components];
                for (&ni, &w) in nn_indices.iter().zip(weights.iter()) {
                    for c in 0..n_components {
                        pos[c] += w * self.embedding[[ni, c]];
                    }
                }
                for c in 0..n_components {
                    pos[c] /= weight_sum.max(1e-10);
                }

                // Refine with SGD
                let n_refine = 30;
                for step in 0..n_refine {
                    let lr = 1.0f32 * (1.0 - step as f32 / n_refine as f32);
                    for (&ni, &w) in nn_indices.iter().zip(weights.iter()) {
                        let mut dist_sq = 0.0f32;
                        let mut disp = [0.0f32; 2];
                        for c in 0..n_components.min(2) {
                            let diff = pos[c] as f32 - self.embedding[[ni, c]] as f32;
                            disp[c] = diff;
                            dist_sq += diff * diff;
                        }
                        dist_sq = dist_sq.max(f32::EPSILON);

                        let pd2b = dist_sq.powf(b);
                        let grad_coeff = (-2.0 * a * b * pd2b) / (dist_sq * (a * pd2b + 1.0));

                        for c in 0..n_components.min(2) {
                            let grad = (grad_coeff * disp[c]).clamp(-4.0, 4.0);
                            pos[c] += (lr * w as f32 * grad) as f64;
                        }
                    }
                }

                pos
            })
            .collect();

        let mut output = Array2::zeros((n_new, n_components));
        for (i, pos) in results.iter().enumerate() {
            for c in 0..n_components {
                output[[i, c]] = pos[c];
            }
        }
        output
    }

    /// Export model as CSV triples (subject, predicate, object)
    pub fn save_triples_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "subject,predicate,object")?;

        let n_train = self.training_data.nrows();
        let n_dims = self.training_data.ncols();
        let n_components = self.embedding.ncols();

        // Model params
        writeln!(f, "model,a,{}", self.a)?;
        writeln!(f, "model,b,{}", self.b)?;
        writeln!(f, "model,n_neighbors,{}", self.n_neighbors)?;
        writeln!(f, "model,n_train,{}", n_train)?;
        writeln!(f, "model,n_dims,{}", n_dims)?;
        writeln!(f, "model,n_components,{}", n_components)?;

        let feat_names: Vec<String> = match &self.feature_names {
            Some(names) => names.clone(),
            None => (0..n_dims).map(|j| format!("f{}", j)).collect(),
        };
        for (j, name) in feat_names.iter().enumerate() {
            writeln!(f, "model,feature_{},{}", j, name)?;
        }

        // Per-point data
        for i in 0..n_train {
            let subj = format!("point_{}", i);
            for c in 0..n_components {
                writeln!(f, "{},umap{},{}", subj, c + 1, self.embedding[[i, c]])?;
            }
            writeln!(f, "{},sigma,{}", subj, self.sigmas[i])?;
            writeln!(f, "{},rho,{}", subj, self.rhos[i])?;
            for j in 0..n_dims {
                writeln!(f, "{},{},{}", subj, feat_names[j], self.training_data[[i, j]])?;
            }
        }

        Ok(())
    }

    /// Load model from CSV triples
    pub fn load_triples_csv(path: &str) -> std::io::Result<Self> {
        use std::io::{BufRead, BufReader};
        let f = std::fs::File::open(path)?;
        let reader = BufReader::new(f);
        let mut triples = Vec::new();
        let mut first = true;
        for line in reader.lines() {
            let line = line?;
            if first { first = false; continue; }
            let parts: Vec<&str> = line.splitn(3, ',').collect();
            if parts.len() == 3 {
                triples.push((parts[0].to_string(), parts[1].to_string(), parts[2].to_string()));
            }
        }
        Ok(Self::from_triples(&triples))
    }

    fn from_triples(triples: &[(String, String, String)]) -> Self {
        let mut params: HashMap<String, String> = HashMap::new();
        let mut point_data: HashMap<String, HashMap<String, String>> = HashMap::new();

        for (s, p, o) in triples {
            if s == "model" {
                params.insert(p.clone(), o.clone());
            } else if s.starts_with("point_") {
                point_data.entry(s.clone()).or_default().insert(p.clone(), o.clone());
            }
        }

        let a: f64 = params["a"].parse().unwrap();
        let b: f64 = params["b"].parse().unwrap();
        let n_neighbors: usize = params["n_neighbors"].parse().unwrap();
        let n_train: usize = params["n_train"].parse().unwrap();
        let n_dims: usize = params["n_dims"].parse().unwrap();
        let n_components: usize = params["n_components"].parse().unwrap();

        let mut feature_names = vec![String::new(); n_dims];
        for j in 0..n_dims {
            if let Some(name) = params.get(&format!("feature_{}", j)) {
                feature_names[j] = name.clone();
            }
        }

        let mut training_data = Array2::zeros((n_train, n_dims));
        let mut embedding = Array2::zeros((n_train, n_components));
        let mut sigmas = vec![0.0; n_train];
        let mut rhos = vec![0.0; n_train];

        for i in 0..n_train {
            if let Some(pdata) = point_data.get(&format!("point_{}", i)) {
                for c in 0..n_components {
                    if let Some(v) = pdata.get(&format!("umap{}", c + 1)) {
                        embedding[[i, c]] = v.parse().unwrap();
                    }
                }
                if let Some(v) = pdata.get("sigma") { sigmas[i] = v.parse().unwrap(); }
                if let Some(v) = pdata.get("rho") { rhos[i] = v.parse().unwrap(); }
                for j in 0..n_dims {
                    if let Some(v) = pdata.get(&feature_names[j]) {
                        training_data[[i, j]] = v.parse().unwrap();
                    }
                }
            }
        }

        Self {
            training_data, embedding, sigmas, rhos,
            a, b, n_neighbors,
            feature_names: Some(feature_names),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_roundtrip() {
        let model = UmapModel {
            training_data: Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            embedding: Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
            sigmas: vec![0.5, 0.6, 0.7],
            rhos: vec![0.1, 0.2, 0.3],
            a: 1.577, b: 0.8951, n_neighbors: 15,
            feature_names: Some(vec!["x".into(), "y".into()]),
        };

        let path = "/tmp/umap_test_model.csv";
        model.save_triples_csv(path).unwrap();
        let restored = UmapModel::load_triples_csv(path).unwrap();

        assert_eq!(restored.a, model.a);
        assert_eq!(restored.n_neighbors, model.n_neighbors);
        assert!((restored.embedding[[0, 0]] - 0.1).abs() < 1e-10);
        assert!((restored.sigmas[1] - 0.6).abs() < 1e-10);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_sample_random() {
        let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect()).unwrap();
        let (indices, train) = UmapModel::sample_train(&data, 0.1, &SamplingStrategy::Random, 42);
        assert_eq!(indices.len(), 10);
        assert_eq!(train.shape(), &[10, 3]);
        // Indices should be sorted and unique
        for i in 1..indices.len() {
            assert!(indices[i] > indices[i - 1]);
        }
    }

    #[test]
    fn test_transform_shape_check() {
        let model = UmapModel {
            training_data: Array2::zeros((5, 3)),
            embedding: Array2::zeros((5, 2)),
            sigmas: vec![1.0; 5],
            rhos: vec![0.0; 5],
            a: 1.577, b: 0.8951, n_neighbors: 3,
            feature_names: Some(vec!["a".into(), "b".into(), "c".into()]),
        };

        let result = std::panic::catch_unwind(|| {
            model.transform(&Array2::zeros((2, 5))) // wrong: 5 features, expects 3
        });
        assert!(result.is_err());
    }
}
