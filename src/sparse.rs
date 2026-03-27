use ndarray::Array2;

/// Compressed Sparse Row (CSR) graph representation.
/// Stores only non-zero edges, reducing memory from O(n²) to O(n*k).
pub struct SparseGraph {
    pub n_nodes: usize,
    /// Row offsets into col_indices/values. Length = n_nodes + 1.
    pub row_offsets: Vec<usize>,
    /// Column indices of non-zero entries.
    pub col_indices: Vec<usize>,
    /// Corresponding edge weights.
    pub values: Vec<f64>,
}

impl SparseGraph {
    /// Build a CSR graph from COO-style triplets (row, col, val).
    /// Entries must not contain duplicates.
    pub fn from_triplets(
        n_nodes: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
    ) -> Self {
        let nnz = rows.len();

        // Sort by (row, col)
        let mut indices: Vec<usize> = (0..nnz).collect();
        indices.sort_by(|&a, &b| {
            rows[a]
                .cmp(&rows[b])
                .then_with(|| cols[a].cmp(&cols[b]))
        });

        let mut sorted_cols = Vec::with_capacity(nnz);
        let mut sorted_vals = Vec::with_capacity(nnz);
        for &i in &indices {
            sorted_cols.push(cols[i]);
            sorted_vals.push(vals[i]);
        }

        // Build row offsets
        let mut row_offsets = vec![0usize; n_nodes + 1];
        for &i in &indices {
            row_offsets[rows[i] + 1] += 1;
        }
        for i in 1..=n_nodes {
            row_offsets[i] += row_offsets[i - 1];
        }

        Self {
            n_nodes,
            row_offsets,
            col_indices: sorted_cols,
            values: sorted_vals,
        }
    }

    /// Number of non-zero edges
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }

    /// Iterate over all edges as (row, col, value)
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.n_nodes).flat_map(move |row| {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            (start..end).map(move |idx| (row, self.col_indices[idx], self.values[idx]))
        })
    }

    /// Get neighbors and weights for a given row
    pub fn row_entries(&self, row: usize) -> (&[usize], &[f64]) {
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        (&self.col_indices[start..end], &self.values[start..end])
    }

    /// Convert to dense Array2 (for spectral initialization on small datasets)
    pub fn to_dense(&self) -> Array2<f64> {
        let mut dense = Array2::zeros((self.n_nodes, self.n_nodes));
        for (row, col, val) in self.edges() {
            dense[[row, col]] = val;
        }
        dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_triplets() {
        let rows = vec![0, 0, 1, 2];
        let cols = vec![1, 2, 0, 0];
        let vals = vec![0.5, 0.3, 0.5, 0.3];

        let graph = SparseGraph::from_triplets(3, &rows, &cols, &vals);
        assert_eq!(graph.n_nodes, 3);
        assert_eq!(graph.nnz(), 4);

        let (cols_0, vals_0) = graph.row_entries(0);
        assert_eq!(cols_0, &[1, 2]);
        assert_eq!(vals_0, &[0.5, 0.3]);
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let rows = vec![0, 1, 1, 2];
        let cols = vec![1, 0, 2, 1];
        let vals = vec![0.8, 0.8, 0.6, 0.6];

        let graph = SparseGraph::from_triplets(3, &rows, &cols, &vals);
        let dense = graph.to_dense();

        assert_eq!(dense[[0, 1]], 0.8);
        assert_eq!(dense[[1, 0]], 0.8);
        assert_eq!(dense[[1, 2]], 0.6);
        assert_eq!(dense[[0, 0]], 0.0);
    }

    #[test]
    fn test_edges_iterator() {
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let vals = vec![0.5, 0.5];

        let graph = SparseGraph::from_triplets(2, &rows, &cols, &vals);
        let edges: Vec<_> = graph.edges().collect();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0], (0, 1, 0.5));
        assert_eq!(edges[1], (1, 0, 0.5));
    }
}
