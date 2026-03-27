#!/usr/bin/env Rscript

# Install uwot if not already installed
if (!require("uwot", quietly = TRUE)) {
  install.packages("uwot")
}

library(uwot)

# Set seed for reproducibility
set.seed(42)

# Read the prepared crabs data
data <- read.csv("data/crabs_data.csv")

cat("=== Running UMAP on Crabs Dataset (R uwot implementation) ===\n\n")
cat("Input data shape:", dim(data), "\n")
cat("Features: FL, RW, CL, CW, BD (5 morphological measurements)\n")
cat("Samples:", nrow(data), "crabs\n")

cat("\nUMAP Parameters:\n")
cat("  n_neighbors: 15\n")
cat("  n_components: 2\n")
cat("  min_dist: 0.1\n")
cat("  learning_rate: 1.0\n")
cat("  n_epochs: 200\n")
cat("  random_state: 42\n")

cat("\nRunning UMAP...\n")

# Run UMAP with same parameters
embedding <- umap(
  data,
  n_neighbors = 15,
  n_components = 2,
  min_dist = 0.1,
  learning_rate = 1.0,
  n_epochs = 200,
  verbose = TRUE,
  ret_model = FALSE
)

cat("\nOutput embedding shape:", dim(embedding), "\n")

# Save to CSV
write.csv(embedding, "uwot_crabs_embedding.csv", row.names = FALSE)

cat("\n=== Statistics ===\n")
cat("Embedding range X:", range(embedding[, 1]), "\n")
cat("Embedding range Y:", range(embedding[, 2]), "\n")
cat("Embedding mean X:", mean(embedding[, 1]), "\n")
cat("Embedding mean Y:", mean(embedding[, 2]), "\n")
cat("Embedding sd X:", sd(embedding[, 1]), "\n")
cat("Embedding sd Y:", sd(embedding[, 2]), "\n")

cat("\nFile saved: uwot_crabs_embedding.csv\n")
