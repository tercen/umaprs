#!/usr/bin/env Rscript

# Final three-way comparison: Custom Rust vs annembed vs uwot

library(ggplot2)
library(gridExtra)

# Read data
data <- read.csv("data/crabs_data.csv")
labels <- read.csv("data/crabs_labels.csv")

cat("=== Three-way UMAP Comparison: Complete ===\n\n")

# Read all embeddings
rust_embedding <- read.csv("rust_crabs_embedding.csv")
uwot_embedding <- read.csv("uwot_crabs_embedding.csv")
annembed_embedding <- read.csv("annembed_crabs_embedding.csv")

cat("Loaded embeddings:\n")
cat("  1. Custom Rust (educational)\n")
cat("  2. annembed (Rust production)\n")
cat("  3. R uwot (reference standard)\n\n")

# Calculate statistics
calc_stats <- function(embedding, name) {
  cat(name, ":\n", sep = "")
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
  cat("  Separation ratio:", sprintf("%.3f", separation_ratio), "\n\n")

  return(list(
    separation_ratio = separation_ratio,
    within_mean = within_mean,
    between_mean = between_mean,
    std_x = sd(embedding$V1),
    std_y = sd(embedding$V2)
  ))
}

cat("=== Statistics and Quality Metrics ===\n\n")
rust_stats <- calc_stats(rust_embedding, "Custom Rust")
annembed_stats <- calc_stats(annembed_embedding, "annembed")
uwot_stats <- calc_stats(uwot_embedding, "R uwot")

# Create visualization
create_plot <- function(embedding, title) {
  df <- data.frame(
    x = embedding$V1,
    y = embedding$V2,
    group = labels$group
  )

  ggplot(df, aes(x = x, y = y, color = group, shape = group)) +
    geom_point(size = 2.5, alpha = 0.7) +
    scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442")) +
    scale_shape_manual(values = c(16, 17, 15, 18)) +
    labs(title = title, x = "UMAP 1", y = "UMAP 2") +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      plot.margin = margin(10, 10, 10, 10)
    ) +
    coord_fixed()
}

p1 <- create_plot(rust_embedding, "Custom Rust UMAP\n(Educational)")
p2 <- create_plot(annembed_embedding, "annembed UMAP\n(Rust Production)")
p3 <- create_plot(uwot_embedding, "R uwot UMAP\n(Reference Standard)")

png("comparison_three_way.png", width = 1800, height = 600, res = 100)
grid.arrange(p1, p2, p3, ncol = 3)
dev.off()

cat("Visualization saved to: comparison_three_way.png\n\n")

# Summary table
cat("=== Summary Comparison Table ===\n\n")
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

cat(sprintf("%-25s %10.3f %12.3f %12.3f %10.3f %10.3f\n",
            "annembed",
            annembed_stats$separation_ratio,
            annembed_stats$within_mean,
            annembed_stats$between_mean,
            annembed_stats$std_x,
            annembed_stats$std_y))

cat(sprintf("%-25s %10.3f %12.3f %12.3f %10.3f %10.3f\n",
            "R uwot",
            uwot_stats$separation_ratio,
            uwot_stats$within_mean,
            uwot_stats$between_mean,
            uwot_stats$std_x,
            uwot_stats$std_y))

cat("\n")

# Quality assessment
cat("=== Quality Assessment vs R uwot (baseline) ===\n\n")

rust_quality <- (rust_stats$separation_ratio / uwot_stats$separation_ratio) * 100
annembed_quality <- (annembed_stats$separation_ratio / uwot_stats$separation_ratio) * 100

cat("Custom Rust quality:", sprintf("%.1f%%", rust_quality), "\n")
cat("annembed quality:", sprintf("%.1f%%", annembed_quality), "\n\n")

# Detailed comparison
cat("=== Detailed Comparison ===\n\n")

cat("Cluster Separation (higher is better):\n")
cat(sprintf("  Custom Rust: %.3f (%s)\n",
    rust_stats$separation_ratio,
    if(rust_stats$separation_ratio > 1.0) "✓ Separated" else "✗ Not separated"))
cat(sprintf("  annembed:    %.3f (%s)\n",
    annembed_stats$separation_ratio,
    if(annembed_stats$separation_ratio > 1.0) "✓ Separated" else "✗ Not separated"))
cat(sprintf("  R uwot:      %.3f (✓ Reference)\n\n", uwot_stats$separation_ratio))

cat("Embedding Compactness (lower std dev is tighter):\n")
cat(sprintf("  Custom Rust: X=%.2f, Y=%.2f (isotropic)\n",
    rust_stats$std_x, rust_stats$std_y))
cat(sprintf("  annembed:    X=%.2f, Y=%.2f (%s)\n",
    annembed_stats$std_x, annembed_stats$std_y,
    if(abs(annembed_stats$std_x - annembed_stats$std_y) < 0.5) "isotropic" else "anisotropic"))
cat(sprintf("  R uwot:      X=%.2f, Y=%.2f (anisotropic - compact)\n\n",
    uwot_stats$std_x, uwot_stats$std_y))

# Rankings
cat("=== Overall Rankings ===\n\n")

implementations <- data.frame(
  Name = c("Custom Rust", "annembed", "R uwot"),
  SepRatio = c(rust_stats$separation_ratio, annembed_stats$separation_ratio, uwot_stats$separation_ratio),
  Quality = c(rust_quality, annembed_quality, 100)
)

implementations <- implementations[order(-implementations$SepRatio),]

for (i in 1:nrow(implementations)) {
  cat(sprintf("%d. %s - %.1f%% quality (separation: %.3f)\n",
      i, implementations$Name[i], implementations$Quality[i], implementations$SepRatio[i]))
}

cat("\n=== Conclusions ===\n\n")

cat("Best Overall:    R uwot (reference standard)\n")
cat("Best Rust:       ",
    if(annembed_stats$separation_ratio > rust_stats$separation_ratio) "annembed" else "Custom Rust",
    "\n")
cat("Best for Learning: Custom Rust (readable, educational)\n")
cat("Best for Production Rust: annembed (HNSW, scalable)\n")

cat("\n=== Comparison Complete ===\n")
