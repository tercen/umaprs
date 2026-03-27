#!/usr/bin/env Rscript

# Three-way comparison: Custom Rust UMAP vs annembed vs uwot

library(uwot)
library(ggplot2)
library(gridExtra)

# Read data
data <- read.csv("data/crabs_data.csv")
labels <- read.csv("data/crabs_labels.csv")

cat("=== Three-way UMAP Comparison ===\n\n")
cat("Dataset: Crabs (200 samples, 5 features)\n")
cat("Groups:", unique(labels$group), "\n\n")

# Common parameters
n_neighbors <- 15
n_components <- 2
min_dist <- 0.1
n_epochs <- 200
set.seed(42)

cat("Common parameters:\n")
cat("  n_neighbors:", n_neighbors, "\n")
cat("  n_components:", n_components, "\n")
cat("  min_dist:", min_dist, "\n")
cat("  n_epochs:", n_epochs, "\n")
cat("  random_state: 42\n\n")

# Read existing embeddings
cat("Reading embeddings...\n")
rust_embedding <- read.csv("rust_crabs_embedding.csv")
uwot_embedding <- read.csv("uwot_crabs_embedding.csv")

cat("  Custom Rust implementation: rust_crabs_embedding.csv\n")
cat("  R uwot implementation: uwot_crabs_embedding.csv\n")

# Try to use umap package (Python UMAP via reticulate) as third comparison
# If not available, we'll just compare the two we have
python_available <- FALSE
python_embedding <- NULL

tryCatch({
  library(reticulate)
  umap_pkg <- import("umap")

  cat("\nRunning Python UMAP (reference implementation)...\n")
  reducer <- umap_pkg$UMAP(
    n_neighbors = as.integer(n_neighbors),
    n_components = as.integer(n_components),
    min_dist = min_dist,
    n_epochs = as.integer(n_epochs),
    random_state = as.integer(42)
  )

  python_embedding <- reducer$fit_transform(as.matrix(data))
  colnames(python_embedding) <- c("V1", "V2")
  python_embedding <- as.data.frame(python_embedding)
  write.csv(python_embedding, "python_umap_crabs_embedding.csv", row.names = FALSE)
  python_available <- TRUE
  cat("  Python UMAP saved to: python_umap_crabs_embedding.csv\n")
}, error = function(e) {
  cat("\nPython UMAP not available (", e$message, ")\n")
  cat("Comparing only Rust and uwot implementations\n\n")
})

# Calculate statistics for all implementations
calc_stats <- function(embedding, name) {
  cat("\n", name, ":\n", sep = "")
  cat("  Range X: [", sprintf("%.3f", min(embedding$V1)), ", ",
      sprintf("%.3f", max(embedding$V1)), "]\n", sep = "")
  cat("  Range Y: [", sprintf("%.3f", min(embedding$V2)), ", ",
      sprintf("%.3f", max(embedding$V2)), "]\n", sep = "")
  cat("  Mean X:", sprintf("%.6f", mean(embedding$V1)), "\n")
  cat("  Mean Y:", sprintf("%.6f", mean(embedding$V2)), "\n")
  cat("  Std X:", sprintf("%.3f", sd(embedding$V1)), "\n")
  cat("  Std Y:", sprintf("%.3f", sd(embedding$V2)), "\n")

  # Calculate clustering metrics
  embedding$group <- labels$group

  within_dist <- c()
  between_dist <- c()

  groups <- unique(labels$group)
  for (g1 in groups) {
    points_g1 <- embedding[embedding$group == g1, c("V1", "V2")]

    # Within-group distances
    if (nrow(points_g1) > 1) {
      dists <- dist(points_g1)
      within_dist <- c(within_dist, as.vector(dists))
    }

    # Between-group distances
    for (g2 in groups) {
      if (g1 < g2) {
        points_g2 <- embedding[embedding$group == g2, c("V1", "V2")]
        for (i in 1:nrow(points_g1)) {
          for (j in 1:nrow(points_g2)) {
            d <- sqrt((points_g1[i,1] - points_g2[j,1])^2 +
                     (points_g1[i,2] - points_g2[j,2])^2)
            between_dist <- c(between_dist, d)
          }
        }
      }
    }
  }

  within_mean <- mean(within_dist)
  within_sd <- sd(within_dist)
  between_mean <- mean(between_dist)
  between_sd <- sd(between_dist)
  separation_ratio <- between_mean / within_mean

  cat("  Within-group distance:", sprintf("%.3f", within_mean),
      "±", sprintf("%.3f", within_sd), "\n")
  cat("  Between-group distance:", sprintf("%.3f", between_mean),
      "±", sprintf("%.3f", between_sd), "\n")
  cat("  Separation ratio:", sprintf("%.3f", separation_ratio), "\n")

  return(list(
    separation_ratio = separation_ratio,
    within_mean = within_mean,
    between_mean = between_mean,
    std_x = sd(embedding$V1),
    std_y = sd(embedding$V2)
  ))
}

cat("\n=== Statistics and Quality Metrics ===")
rust_stats <- calc_stats(rust_embedding, "Custom Rust UMAP")
uwot_stats <- calc_stats(uwot_embedding, "R uwot UMAP")

if (python_available) {
  python_stats <- calc_stats(python_embedding, "Python UMAP (reference)")
}

# Create comparison visualization
create_plot <- function(embedding, title) {
  df <- data.frame(
    x = embedding$V1,
    y = embedding$V2,
    group = labels$group
  )

  ggplot(df, aes(x = x, y = y, color = group, shape = group)) +
    geom_point(size = 2, alpha = 0.7) +
    scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442")) +
    scale_shape_manual(values = c(16, 17, 15, 18)) +
    labs(title = title, x = "UMAP 1", y = "UMAP 2") +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    coord_fixed()
}

if (python_available) {
  # Three-way comparison
  p1 <- create_plot(rust_embedding, "Custom Rust UMAP\n(This Implementation)")
  p2 <- create_plot(python_embedding, "Python UMAP\n(Reference Implementation)")
  p3 <- create_plot(uwot_embedding, "R uwot UMAP\n(Production R Package)")

  png("comparison_three_way.png", width = 1800, height = 600, res = 100)
  grid.arrange(p1, p2, p3, ncol = 3)
  dev.off()

  cat("\n\nVisualization saved to: comparison_three_way.png\n")
} else {
  # Two-way comparison (already exists)
  cat("\n\nUsing existing two-way comparison: comparison_crabs.png\n")
}

# Summary table
cat("\n\n=== Summary Comparison Table ===\n\n")
cat(sprintf("%-25s %10s %12s %12s %10s %10s\n",
            "Implementation", "Sep.Ratio", "Within-Dist", "Between-Dist", "Std-X", "Std-Y"))
cat(paste(rep("-", 85), collapse = ""), "\n")

cat(sprintf("%-25s %10.3f %12.3f %12.3f %10.3f %10.3f\n",
            "Custom Rust",
            rust_stats$separation_ratio,
            rust_stats$within_mean,
            rust_stats$between_mean,
            rust_stats$std_x,
            rust_stats$std_y))

if (python_available) {
  cat(sprintf("%-25s %10.3f %12.3f %12.3f %10.3f %10.3f\n",
              "Python UMAP (reference)",
              python_stats$separation_ratio,
              python_stats$within_mean,
              python_stats$between_mean,
              python_stats$std_x,
              python_stats$std_y))
}

cat(sprintf("%-25s %10.3f %12.3f %12.3f %10.3f %10.3f\n",
            "R uwot",
            uwot_stats$separation_ratio,
            uwot_stats$within_mean,
            uwot_stats$between_mean,
            uwot_stats$std_x,
            uwot_stats$std_y))

cat("\n")
cat("Legend:\n")
cat("  Sep.Ratio  = Between-group distance / Within-group distance (higher is better)\n")
cat("  Within-Dist = Average distance between points in same group (lower is better)\n")
cat("  Between-Dist = Average distance between different groups (higher is better)\n")
cat("  Std-X/Y    = Standard deviation in each dimension (reflects spread)\n")

cat("\n\n=== Quality Assessment ===\n\n")

rust_quality <- (rust_stats$separation_ratio / uwot_stats$separation_ratio) * 100
cat("Custom Rust quality vs uwot:", sprintf("%.1f%%", rust_quality), "\n")

if (python_available) {
  cat("Python UMAP quality vs uwot:",
      sprintf("%.1f%%", (python_stats$separation_ratio / uwot_stats$separation_ratio) * 100), "\n")
}

cat("\n")
if (rust_stats$separation_ratio > 1.0) {
  cat("✓ Custom Rust achieves cluster separation (ratio > 1.0)\n")
} else {
  cat("✗ Custom Rust does NOT achieve cluster separation (ratio < 1.0)\n")
}

if (uwot_stats$separation_ratio > 1.0) {
  cat("✓ R uwot achieves cluster separation (ratio > 1.0)\n")
}

if (python_available && python_stats$separation_ratio > 1.0) {
  cat("✓ Python UMAP achieves cluster separation (ratio > 1.0)\n")
}

cat("\n=== Comparison Complete ===\n")
