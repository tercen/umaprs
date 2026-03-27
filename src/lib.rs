use ndarray::Array2;

pub mod gpu;
mod hnsw;
mod kdtree;
mod knn;
mod fuzzy;
mod model;
mod quantize;
mod sparse;
mod spectral;
mod optimize;

pub use knn::{compute_knn_graph, compute_knn_bruteforce, compute_knn_quant_hnsw, compute_knn_quant_hnsw_8bit, compute_knn_hnsw_f32, compute_knn_quant4_kdtree, compute_knn_quant8_kdtree};
pub use fuzzy::compute_fuzzy_simplicial_set;
pub use spectral::{spectral_layout, spectral_layout_with_data};
pub use optimize::optimize_layout;
pub use sparse::SparseGraph;
pub use model::{UmapModel, SamplingStrategy};

/// Initialization method for the embedding
#[derive(Clone, Debug)]
pub enum InitMethod {
    Spectral,
    Pca,
    Random,
    Auto,
}

/// Distance metric
#[derive(Clone, Debug)]
pub enum Metric {
    Euclidean,
}

/// kNN search method
#[derive(Clone, Debug)]
pub enum KnnMethod {
    Auto,
    BruteForce,
    KdTree,
    Hnsw,
    TurboQuant4KdTree,
    TurboQuant8KdTree,
    TurboQuant4Hnsw,
    TurboQuant8Hnsw,
    /// GPU brute-force via cuBLAS (requires 'cuda' feature)
    Gpu,
}

/// Model export format
#[derive(Clone, Debug)]
pub enum ModelFormat {
    /// No model export — just return the embedding
    None,
    /// Export as CSV triples (subject, predicate, object)
    Csv(String),
}

/// UMAP dimensionality reduction
pub struct UMAP {
    pub n_neighbors: usize,
    pub n_components: usize,
    pub min_dist: f64,
    pub spread: f64,
    pub learning_rate: f64,
    pub n_epochs: usize,
    pub negative_sample_rate: f64,
    pub repulsion_strength: f64,
    pub init: InitMethod,
    pub metric: Metric,
    pub knn_method: KnnMethod,
    pub pca: Option<usize>,
    pub random_state: Option<u64>,
    pub model_format: ModelFormat,
    pub feature_names: Option<Vec<String>>,
    /// Fraction of data to use for training (default: None = use all).
    /// When set (e.g., 0.1), fit on a random subset and transform the rest.
    pub train_size: Option<f64>,
    /// Sampling strategy for training subset (default: Random)
    pub sampling: SamplingStrategy,
}

impl Default for UMAP {
    fn default() -> Self {
        UMAP {
            n_neighbors: 15,
            n_components: 2,
            min_dist: 0.1,
            spread: 1.0,
            learning_rate: 1.0,
            n_epochs: 0,
            negative_sample_rate: 5.0,
            repulsion_strength: 1.0,
            init: InitMethod::Auto,
            metric: Metric::Euclidean,
            knn_method: KnnMethod::Auto,
            pca: None,
            random_state: None,
            model_format: ModelFormat::None,
            feature_names: None,
            train_size: None,
            sampling: SamplingStrategy::Random,
        }
    }
}

impl UMAP {
    pub fn new() -> Self { Self::default() }

    pub fn n_neighbors(mut self, v: usize) -> Self { self.n_neighbors = v; self }
    pub fn n_components(mut self, v: usize) -> Self { self.n_components = v; self }
    pub fn min_dist(mut self, v: f64) -> Self { self.min_dist = v; self }
    pub fn spread(mut self, v: f64) -> Self { self.spread = v; self }
    pub fn learning_rate(mut self, v: f64) -> Self { self.learning_rate = v; self }
    pub fn n_epochs(mut self, v: usize) -> Self { self.n_epochs = v; self }
    pub fn negative_sample_rate(mut self, v: f64) -> Self { self.negative_sample_rate = v; self }
    pub fn repulsion_strength(mut self, v: f64) -> Self { self.repulsion_strength = v; self }
    pub fn init(mut self, v: InitMethod) -> Self { self.init = v; self }
    pub fn metric(mut self, v: Metric) -> Self { self.metric = v; self }
    pub fn knn_method(mut self, v: KnnMethod) -> Self { self.knn_method = v; self }
    pub fn pca(mut self, v: usize) -> Self { self.pca = Some(v); self }
    pub fn random_state(mut self, v: u64) -> Self { self.random_state = Some(v); self }
    pub fn model_format(mut self, v: ModelFormat) -> Self { self.model_format = v; self }
    pub fn feature_names(mut self, v: Vec<String>) -> Self { self.feature_names = Some(v); self }
    pub fn train_size(mut self, v: f64) -> Self { self.train_size = Some(v); self }
    pub fn sampling(mut self, v: SamplingStrategy) -> Self { self.sampling = v; self }

    fn resolve_n_epochs(&self, n_samples: usize) -> usize {
        if self.n_epochs > 0 { self.n_epochs }
        else if n_samples < 10000 { 500 }
        else { 200 }
    }

    fn find_ab_params(&self) -> (f64, f64) {
        optimize::find_ab_params(self.min_dist, self.spread)
    }

    fn prepare_data<'a>(&self, data: &'a Array2<f64>, reduced: &'a mut Option<Array2<f64>>) -> &'a Array2<f64> {
        if let Some(pca_dims) = self.pca {
            let n = data.nrows();
            let d = data.ncols();
            if pca_dims < d && pca_dims < n {
                eprintln!("PCA reduction: {} -> {} dims", d, pca_dims);
                *reduced = Some(spectral::pca_reduce(data, pca_dims));
                return reduced.as_ref().unwrap();
            }
        }
        data
    }

    fn compute_knn(&self, data: &Array2<f64>) -> Array2<usize> {
        let k = self.n_neighbors;
        match &self.knn_method {
            KnnMethod::Auto => compute_knn_graph(data, k),
            KnnMethod::BruteForce => compute_knn_bruteforce(data, k),
            KnnMethod::KdTree => knn::compute_knn_kdtree(data, k),
            KnnMethod::Hnsw => compute_knn_hnsw_f32(data, k),
            KnnMethod::TurboQuant4KdTree => compute_knn_quant4_kdtree(data, k),
            KnnMethod::TurboQuant8KdTree => compute_knn_quant8_kdtree(data, k),
            KnnMethod::TurboQuant4Hnsw => compute_knn_quant_hnsw(data, k),
            KnnMethod::TurboQuant8Hnsw => compute_knn_quant_hnsw_8bit(data, k),
            KnnMethod::Gpu => gpu::compute_knn_gpu(data, k),
        }
    }

    /// Fit and return embedding only.
    /// If `train_size` is set, fits on a subset and transforms the rest.
    pub fn fit_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut reduced = None;
        let work_data = self.prepare_data(data, &mut reduced);

        match self.train_size {
            Some(frac) if frac < 1.0 => {
                let seed = self.random_state.unwrap_or(42);
                let (train_idx, train_data) = UmapModel::sample_train(work_data, frac, &self.sampling, seed);
                let n_total = work_data.nrows();
                let n_train = train_idx.len();

                eprintln!("Training on {} / {} samples ({:.0}%), transforming the rest",
                    n_train, n_total, frac * 100.0);

                // Fit on training subset
                let knn_indices = self.compute_knn(&train_data);
                let (train_emb, model) = self.fit_internal(&train_data, &knn_indices);
                let model = model.unwrap();

                // Collect non-training indices
                let mut is_train = vec![false; n_total];
                for &i in &train_idx { is_train[i] = true; }

                let rest_idx: Vec<usize> = (0..n_total).filter(|i| !is_train[*i]).collect();

                // Build rest data matrix
                let n_dims = work_data.ncols();
                let mut rest_data = Array2::zeros((rest_idx.len(), n_dims));
                for (i, &orig) in rest_idx.iter().enumerate() {
                    for j in 0..n_dims {
                        rest_data[[i, j]] = work_data[[orig, j]];
                    }
                }

                // Transform rest
                eprintln!("Transforming {} remaining samples", rest_idx.len());
                let rest_emb = model.transform(&rest_data);

                // Merge into original order
                let n_components = train_emb.ncols();
                let mut output = Array2::zeros((n_total, n_components));
                for (i, &orig) in train_idx.iter().enumerate() {
                    for c in 0..n_components {
                        output[[orig, c]] = train_emb[[i, c]];
                    }
                }
                for (i, &orig) in rest_idx.iter().enumerate() {
                    for c in 0..n_components {
                        output[[orig, c]] = rest_emb[[i, c]];
                    }
                }

                // Save model if requested
                if let ModelFormat::Csv(path) = &self.model_format {
                    model.save_triples_csv(path).expect("Failed to save model CSV");
                    eprintln!("Model saved to: {}", path);
                }

                output
            }
            _ => {
                let knn_indices = self.compute_knn(work_data);
                self.fit_transform_with_knn(work_data, &knn_indices)
            }
        }
    }

    /// Fit using pre-computed kNN, return embedding only
    pub fn fit_transform_with_knn(&self, data: &Array2<f64>, knn_indices: &Array2<usize>) -> Array2<f64> {
        let (embedding, _) = self.fit_internal(data, knn_indices);
        embedding
    }

    /// Fit and return both embedding and model (for transform)
    pub fn fit(&self, data: &Array2<f64>) -> (Array2<f64>, UmapModel) {
        let mut reduced = None;
        let work_data = self.prepare_data(data, &mut reduced);
        let knn_indices = self.compute_knn(work_data);
        let (embedding, model) = self.fit_internal(work_data, &knn_indices);

        let mut model = model.unwrap();

        // Export if requested
        if let ModelFormat::Csv(path) = &self.model_format {
            model.save_triples_csv(path).expect("Failed to save model CSV");
            eprintln!("Model saved to: {}", path);
        }

        (embedding, model)
    }

    fn fit_internal(&self, data: &Array2<f64>, knn_indices: &Array2<usize>) -> (Array2<f64>, Option<UmapModel>) {
        let n_samples = data.nrows();
        let n_epochs = self.resolve_n_epochs(n_samples);
        let (a, b) = self.find_ab_params();

        // Fuzzy simplicial set (with sigmas/rhos)
        let fuzzy_result = fuzzy::compute_fuzzy_simplicial_set_full(knn_indices, data, self.n_neighbors);

        // Initialize embedding
        let mut embedding = match &self.init {
            InitMethod::Spectral => spectral_layout(&fuzzy_result.graph, self.n_components, self.random_state),
            InitMethod::Pca => spectral::pca_initialization(data, self.n_components, self.random_state),
            InitMethod::Random => spectral::random_initialization(fuzzy_result.graph.n_nodes, self.n_components, self.random_state),
            InitMethod::Auto => spectral_layout_with_data(
                &fuzzy_result.graph, self.n_components, self.random_state, Some(data),
            ),
        };

        // Optimize
        optimize_layout(
            &mut embedding,
            &fuzzy_result.graph,
            n_epochs,
            self.learning_rate,
            self.min_dist,
            self.spread,
            self.negative_sample_rate,
            self.repulsion_strength,
            self.random_state,
        );

        // Build model
        let model = UmapModel {
            training_data: data.clone(),
            embedding: embedding.clone(),
            sigmas: fuzzy_result.sigmas,
            rhos: fuzzy_result.rhos,
            a,
            b,
            n_neighbors: self.n_neighbors,
            feature_names: self.feature_names.clone(),
        };

        (embedding, Some(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_umap_basic() {
        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();
        let umap = UMAP::new().n_neighbors(5).n_epochs(10);
        let embedding = umap.fit_transform(&data);
        assert_eq!(embedding.shape(), &[10, 2]);
    }

    #[test]
    fn test_umap_params() {
        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();
        let umap = UMAP::new()
            .n_neighbors(5)
            .n_epochs(10)
            .min_dist(0.5)
            .spread(2.0)
            .negative_sample_rate(3.0)
            .repulsion_strength(1.5)
            .init(InitMethod::Random)
            .random_state(42);
        let embedding = umap.fit_transform(&data);
        assert_eq!(embedding.shape(), &[10, 2]);
    }

    #[test]
    fn test_fit_model() {
        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();
        let umap = UMAP::new().n_neighbors(5).n_epochs(10).random_state(42);
        let (embedding, model) = umap.fit(&data);
        assert_eq!(embedding.shape(), &[10, 2]);
        assert_eq!(model.training_data.shape(), &[10, 5]);
        assert_eq!(model.sigmas.len(), 10);
        assert_eq!(model.rhos.len(), 10);
    }

    #[test]
    fn test_fit_transform_model() {
        let data = Array2::from_shape_vec((20, 4), (0..80).map(|x| x as f64).collect()).unwrap();
        let umap = UMAP::new().n_neighbors(5).n_epochs(10).random_state(42);
        let (_, model) = umap.fit(&data);

        // Transform new data
        let new_data = Array2::from_shape_vec((5, 4), (0..20).map(|x| x as f64 + 0.5).collect()).unwrap();
        let new_emb = model.transform(&new_data);
        assert_eq!(new_emb.shape(), &[5, 2]);
    }
}
