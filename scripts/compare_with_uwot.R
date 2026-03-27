#!/usr/bin/env Rscript

# Install uwot if not already installed
if (!require("uwot", quietly = TRUE)) {
  install.packages("uwot")
}

library(uwot)

# Set seed for reproducibility
set.seed(42)

# Create same test data as in our Rust implementation
# 100 samples, 10 features
n_samples <- 100
n_features <- 10

data <- matrix(0, nrow = n_samples, ncol = n_features)
for (i in 1:n_samples) {
  for (j in 1:n_features) {
    data[i, j] <- (i - 1) * (j - 1) + sin(i - 1)
  }
}

cat("Input data shape:", dim(data), "\n")
cat("First 5 rows of input data:\n")
print(head(data, 5))

cat("\n=== Running UMAP with uwot ===\n")

# Run UMAP with similar parameters to our Rust implementation
embedding <- umap(
  data,
  n_neighbors = 15,
  n_components = 2,
  min_dist = 0.1,
  learning_rate = 1.0,
  n_epochs = 100,
  verbose = TRUE,
  ret_model = FALSE
)

cat("\nOutput embedding shape:", dim(embedding), "\n")
cat("\nFirst 10 embedded points from uwot:\n")
print(head(embedding, 10))

# Save to CSV for comparison
write.csv(data, "test_data.csv", row.names = FALSE)
write.csv(embedding, "uwot_embedding.csv", row.names = FALSE)

cat("\n=== Statistics ===\n")
cat("Embedding range X:", range(embedding[, 1]), "\n")
cat("Embedding range Y:", range(embedding[, 2]), "\n")
cat("Embedding mean X:", mean(embedding[, 1]), "\n")
cat("Embedding mean Y:", mean(embedding[, 2]), "\n")
cat("Embedding sd X:", sd(embedding[, 1]), "\n")
cat("Embedding sd Y:", sd(embedding[, 2]), "\n")

cat("\nFiles saved: test_data.csv, uwot_embedding.csv\n")
