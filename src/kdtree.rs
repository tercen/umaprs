/// Simple kd-tree for exact kNN search.
/// Works well up to ~40 dimensions (like FNN/uwot).

pub struct KdTree {
    nodes: Vec<KdNode>,
    data: Vec<f32>,   // flat: [x0_d0, x0_d1, ..., x1_d0, ...]
    n_dims: usize,
}

struct KdNode {
    point_idx: u32,
    split_dim: u16,
    left: u32,   // 0 = no child
    right: u32,
}

impl KdTree {
    /// Build kd-tree from flat f32 data (n_points × n_dims).
    pub fn build(data: &[f32], n_points: usize, n_dims: usize) -> Self {
        let mut indices: Vec<u32> = (0..n_points as u32).collect();
        let mut nodes = Vec::with_capacity(n_points);
        // Reserve index 0 as "null"
        nodes.push(KdNode { point_idx: 0, split_dim: 0, left: 0, right: 0 });

        build_recursive(&mut nodes, &mut indices, data, n_dims, 0);

        Self { nodes, data: data.to_vec(), n_dims }
    }

    /// Find k nearest neighbors of point q. Returns (index, distance) pairs sorted by distance.
    pub fn knn(&self, q_idx: usize, k: usize) -> Vec<(u32, f32)> {
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        self.search_recursive(1, &self.data[q_idx * self.n_dims..], q_idx as u32, k, &mut best);
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        best.truncate(k);
        best
    }

    fn search_recursive(
        &self,
        node_idx: u32,
        query: &[f32],
        q_idx: u32,
        k: usize,
        best: &mut Vec<(u32, f32)>,
    ) {
        if node_idx == 0 { return; }
        let node = &self.nodes[node_idx as usize];
        let point = &self.data[node.point_idx as usize * self.n_dims..][..self.n_dims];

        // Distance to this point
        if node.point_idx != q_idx {
            let dist = dist_sq(query, point);
            let worst = if best.len() >= k {
                best.iter().map(|&(_, d)| d).fold(f32::NEG_INFINITY, f32::max)
            } else {
                f32::MAX
            };

            if best.len() < k || dist < worst {
                if best.len() >= k {
                    // Remove worst
                    let worst_idx = best.iter().enumerate()
                        .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
                        .map(|(i, _)| i).unwrap();
                    best.swap_remove(worst_idx);
                }
                best.push((node.point_idx, dist));
            }
        }

        let split = node.split_dim as usize;
        let diff = query[split] - point[split];

        // Visit nearer side first
        let (first, second) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        self.search_recursive(first, query, q_idx, k, best);

        // Check if we need to visit the other side
        let worst = if best.len() >= k {
            best.iter().map(|&(_, d)| d).fold(f32::NEG_INFINITY, f32::max)
        } else {
            f32::MAX
        };

        if diff * diff < worst || best.len() < k {
            self.search_recursive(second, query, q_idx, k, best);
        }
    }
}

fn build_recursive(
    nodes: &mut Vec<KdNode>,
    indices: &mut [u32],
    data: &[f32],
    n_dims: usize,
    depth: usize,
) -> u32 {
    if indices.is_empty() { return 0; }

    let split_dim = depth % n_dims;

    // Find median by sorting on split dimension
    indices.sort_unstable_by(|&a, &b| {
        let va = data[a as usize * n_dims + split_dim];
        let vb = data[b as usize * n_dims + split_dim];
        va.partial_cmp(&vb).unwrap()
    });

    let mid = indices.len() / 2;
    let point_idx = indices[mid];

    let node_idx = nodes.len() as u32;
    nodes.push(KdNode { point_idx, split_dim: split_dim as u16, left: 0, right: 0 });

    let (left_slice, right_slice) = indices.split_at_mut(mid);
    // right_slice[0] is the median point, skip it
    let right_slice = if right_slice.len() > 1 { &mut right_slice[1..] } else { &mut [] };

    let left = build_recursive(nodes, left_slice, data, n_dims, depth + 1);
    let right = build_recursive(nodes, right_slice, data, n_dims, depth + 1);

    nodes[node_idx as usize].left = left;
    nodes[node_idx as usize].right = right;

    node_idx
}

#[inline]
fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_basic() {
        // 2D points: two clusters
        let data: Vec<f32> = vec![
            0.0, 0.0,  0.1, 0.0,  0.0, 0.1,  0.1, 0.1,
            10.0, 10.0,  10.1, 10.0,  10.0, 10.1,  10.1, 10.1,
        ];
        let tree = KdTree::build(&data, 8, 2);

        let results = tree.knn(0, 3);
        assert_eq!(results.len(), 3);
        for &(nb, _) in &results {
            assert!(nb < 4, "point 0 neighbor {} should be in cluster A", nb);
        }

        let results = tree.knn(4, 3);
        for &(nb, _) in &results {
            assert!(nb >= 4, "point 4 neighbor {} should be in cluster B", nb);
        }
    }

    #[test]
    fn test_kdtree_exact() {
        // Verify exact results match brute-force
        let data: Vec<f32> = vec![
            0.0, 0.0,  3.0, 0.0,  1.0, 0.0,  2.0, 0.0,  4.0, 0.0,
        ];
        let tree = KdTree::build(&data, 5, 2);

        // Point 0 (0,0): nearest should be 2 (1,0), then 3 (2,0)
        let results = tree.knn(0, 2);
        assert_eq!(results[0].0, 2); // (1,0)
        assert_eq!(results[1].0, 3); // (2,0)
    }
}
