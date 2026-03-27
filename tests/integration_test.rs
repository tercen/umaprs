use ndarray::Array2;
use umaprs::UMAP;

#[test]
fn test_umap_basic_transform() {
    // Create simple 2D data that should embed nicely
    let data = Array2::from_shape_vec(
        (10, 5),
        (0..50).map(|x| x as f64).collect()
    ).unwrap();

    let umap = UMAP::new()
        .n_neighbors(5)
        .n_components(2)
        .n_epochs(50)
        .random_state(42);

    let embedding = umap.fit_transform(&data);

    assert_eq!(embedding.shape(), &[10, 2]);
    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_umap_different_dimensions() {
    let data = Array2::from_shape_vec(
        (20, 10),
        (0..200).map(|x| (x as f64) / 10.0).collect()
    ).unwrap();

    // Test 3D embedding
    let umap = UMAP::new()
        .n_neighbors(5)
        .n_components(3)
        .n_epochs(20)
        .random_state(42);

    let embedding = umap.fit_transform(&data);

    assert_eq!(embedding.shape(), &[20, 3]);
    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_umap_reproducibility() {
    let data = Array2::from_shape_vec(
        (15, 8),
        (0..120).map(|x| x as f64).collect()
    ).unwrap();

    let umap1 = UMAP::new()
        .n_neighbors(5)
        .n_epochs(30)
        .random_state(123);

    let umap2 = UMAP::new()
        .n_neighbors(5)
        .n_epochs(30)
        .random_state(123);

    let embedding1 = umap1.fit_transform(&data);
    let embedding2 = umap2.fit_transform(&data);

    // With same random seed, results should be identical
    assert_eq!(embedding1.shape(), embedding2.shape());
    for (a, b) in embedding1.iter().zip(embedding2.iter()) {
        assert!((a - b).abs() < 1e-10, "Embeddings differ: {} vs {}", a, b);
    }
}

#[test]
fn test_umap_builder_pattern() {
    let data = Array2::from_shape_vec(
        (10, 5),
        vec![1.0; 50]
    ).unwrap();

    let umap = UMAP::new()
        .n_neighbors(10)
        .n_components(2)
        .min_dist(0.05)
        .learning_rate(0.5)
        .n_epochs(50)
        .random_state(42);

    let embedding = umap.fit_transform(&data);

    assert_eq!(embedding.shape(), &[10, 2]);
}

#[test]
fn test_umap_clusters() {
    // Create two well-separated clusters
    let mut data_vec = Vec::new();

    // Cluster 1: points near origin
    for i in 0..10 {
        for j in 0..5 {
            data_vec.push((i as f64) * 0.1 + (j as f64) * 0.05);
        }
    }

    // Cluster 2: points far from origin
    for i in 0..10 {
        for j in 0..5 {
            data_vec.push(100.0 + (i as f64) * 0.1 + (j as f64) * 0.05);
        }
    }

    let data = Array2::from_shape_vec((20, 5), data_vec).unwrap();

    let umap = UMAP::new()
        .n_neighbors(5)
        .n_components(2)
        .n_epochs(100)
        .random_state(42);

    let embedding = umap.fit_transform(&data);

    assert_eq!(embedding.shape(), &[20, 2]);
    assert!(embedding.iter().all(|&x| x.is_finite()));

    // Check that clusters are somewhat separated in embedding
    let cluster1_mean = (0..10)
        .map(|i| (embedding[[i, 0]] + embedding[[i, 1]]) / 2.0)
        .sum::<f64>() / 10.0;

    let cluster2_mean = (10..20)
        .map(|i| (embedding[[i, 0]] + embedding[[i, 1]]) / 2.0)
        .sum::<f64>() / 10.0;

    // Clusters should have different mean positions
    assert!((cluster1_mean - cluster2_mean).abs() > 0.1);
}

#[test]
fn test_umap_single_component() {
    let data = Array2::from_shape_vec(
        (15, 10),
        (0..150).map(|x| x as f64).collect()
    ).unwrap();

    let umap = UMAP::new()
        .n_neighbors(5)
        .n_components(1)
        .n_epochs(20)
        .random_state(42);

    let embedding = umap.fit_transform(&data);

    assert_eq!(embedding.shape(), &[15, 1]);
    assert!(embedding.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_umap_larger_dataset() {
    let data = Array2::from_shape_vec(
        (50, 20),
        (0..1000).map(|x| (x as f64).sin()).collect()
    ).unwrap();

    let umap = UMAP::new()
        .n_neighbors(10)
        .n_components(2)
        .n_epochs(50)
        .random_state(42);

    let embedding = umap.fit_transform(&data);

    assert_eq!(embedding.shape(), &[50, 2]);
    assert!(embedding.iter().all(|&x| x.is_finite()));
}
