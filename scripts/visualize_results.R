#!/usr/bin/env Rscript

# Install ggplot2 if not available
if (!require("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}

library(ggplot2)

cat("=== UMAP Results Visualization ===\n\n")

# Read the Rust UMAP results
if (file.exists("rust_embedding.csv")) {
  cat("Reading Rust embedding results...\n")
  embedding <- read.csv("rust_embedding.csv")

  # Add sample indices as groups (for coloring)
  # Let's create groups based on sample index to see structure
  embedding$sample_id <- 1:nrow(embedding)
  embedding$group <- cut(embedding$sample_id,
                         breaks = 5,
                         labels = c("Group 1", "Group 2", "Group 3", "Group 4", "Group 5"))

  cat(paste("Loaded", nrow(embedding), "samples\n\n"))

  # Print summary statistics
  cat("Summary Statistics:\n")
  cat(paste("V1 range: [", round(min(embedding$V1), 2), ",", round(max(embedding$V1), 2), "]\n"))
  cat(paste("V2 range: [", round(min(embedding$V2), 2), ",", round(max(embedding$V2), 2), "]\n"))
  cat(paste("V1 mean:", round(mean(embedding$V1), 2), "sd:", round(sd(embedding$V1), 2), "\n"))
  cat(paste("V2 mean:", round(mean(embedding$V2), 2), "sd:", round(sd(embedding$V2), 2), "\n\n"))

  # Create visualization
  cat("Generating visualizations...\n\n")

  # Plot 1: Colored by sample group
  p1 <- ggplot(embedding, aes(x = V1, y = V2, color = group)) +
    geom_point(size = 3, alpha = 0.7) +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = "UMAP Embedding (Rust Implementation)",
      subtitle = "Colored by sample groups (quintiles)",
      x = "UMAP 1",
      y = "UMAP 2",
      color = "Sample Group"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "right"
    )

  # Plot 2: Colored by sample ID (continuous)
  p2 <- ggplot(embedding, aes(x = V1, y = V2, color = sample_id)) +
    geom_point(size = 3, alpha = 0.7) +
    scale_color_viridis_c() +
    labs(
      title = "UMAP Embedding (Rust Implementation)",
      subtitle = "Colored by sample index",
      x = "UMAP 1",
      y = "UMAP 2",
      color = "Sample ID"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "right"
    )

  # Plot 3: Simple scatter with density contours
  p3 <- ggplot(embedding, aes(x = V1, y = V2)) +
    geom_density_2d(alpha = 0.3, color = "blue") +
    geom_point(size = 2, alpha = 0.6, color = "darkred") +
    labs(
      title = "UMAP Embedding with Density Contours",
      subtitle = "Rust implementation results",
      x = "UMAP 1",
      y = "UMAP 2"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12)
    )

  # Save plots
  ggsave("umap_plot_groups.png", plot = p1, width = 10, height = 7, dpi = 300)
  cat("Saved: umap_plot_groups.png\n")

  ggsave("umap_plot_continuous.png", plot = p2, width = 10, height = 7, dpi = 300)
  cat("Saved: umap_plot_continuous.png\n")

  ggsave("umap_plot_density.png", plot = p3, width = 10, height = 7, dpi = 300)
  cat("Saved: umap_plot_density.png\n")

  # Display first plot (if running interactively)
  print(p1)

  cat("\nVisualization complete!\n")
  cat("Generated files:\n")
  cat("  - umap_plot_groups.png (colored by groups)\n")
  cat("  - umap_plot_continuous.png (colored by sample ID)\n")
  cat("  - umap_plot_density.png (with density contours)\n")

} else {
  cat("ERROR: rust_embedding.csv not found!\n")
  cat("Please run: cargo run --example compare\n")
  cat("This will generate the rust_embedding.csv file.\n")
  quit(status = 1)
}
