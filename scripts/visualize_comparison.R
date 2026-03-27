#!/usr/bin/env Rscript

library(ggplot2)
library(gridExtra)

# Read embeddings
rust_emb <- read.csv("rust_crabs_embedding.csv")
uwot_emb <- read.csv("uwot_crabs_embedding.csv")

# Read labels
labels <- read.csv("data/crabs_labels.csv")

# Create dataframes for plotting
rust_df <- data.frame(
  UMAP1 = rust_emb$V1,
  UMAP2 = rust_emb$V2,
  species = labels$sp,
  sex = labels$sex,
  group = labels$group
)

uwot_df <- data.frame(
  UMAP1 = uwot_emb$V1,
  UMAP2 = uwot_emb$V2,
  species = labels$sp,
  sex = labels$sex,
  group = labels$group
)

# Define colors
colors <- c("B_F" = "#E69F00", "B_M" = "#56B4E9",
            "O_F" = "#009E73", "O_M" = "#F0E442")

# Create plots
p1 <- ggplot(rust_df, aes(x = UMAP1, y = UMAP2, color = group, shape = group)) +
  geom_point(size = 2.5, alpha = 0.7) +
  scale_color_manual(values = colors,
                     labels = c("Blue Female", "Blue Male", "Orange Female", "Orange Male")) +
  scale_shape_manual(values = c(16, 17, 15, 18),
                     labels = c("Blue Female", "Blue Male", "Orange Female", "Orange Male")) +
  labs(title = "Rust UMAP Implementation",
       subtitle = "Crabs Dataset (200 samples, 5 features)",
       x = "UMAP 1",
       y = "UMAP 2",
       color = "Group",
       shape = "Group") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10))

p2 <- ggplot(uwot_df, aes(x = UMAP1, y = UMAP2, color = group, shape = group)) +
  geom_point(size = 2.5, alpha = 0.7) +
  scale_color_manual(values = colors,
                     labels = c("Blue Female", "Blue Male", "Orange Female", "Orange Male")) +
  scale_shape_manual(values = c(16, 17, 15, 18),
                     labels = c("Blue Female", "Blue Male", "Orange Female", "Orange Male")) +
  labs(title = "R uwot Implementation",
       subtitle = "Crabs Dataset (200 samples, 5 features)",
       x = "UMAP 1",
       y = "UMAP 2",
       color = "Group",
       shape = "Group") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10))

# Save combined plot
png("comparison_crabs.png", width = 14, height = 6, units = "in", res = 300)
grid.arrange(p1, p2, ncol = 2)
dev.off()

cat("Visualization saved to: comparison_crabs.png\n")

# Calculate clustering metrics
calc_separation <- function(df) {
  # Calculate within-group and between-group distances
  groups <- unique(df$group)
  within_dist <- numeric(0)
  between_dist <- numeric(0)

  for (g1 in groups) {
    group1 <- df[df$group == g1, c("UMAP1", "UMAP2")]

    # Within-group distances
    if (nrow(group1) > 1) {
      dists <- dist(group1)
      within_dist <- c(within_dist, as.numeric(dists))
    }

    # Between-group distances
    for (g2 in groups) {
      if (g1 < g2) {
        group2 <- df[df$group == g2, c("UMAP1", "UMAP2")]
        for (i in 1:nrow(group1)) {
          for (j in 1:nrow(group2)) {
            d <- sqrt(sum((group1[i,] - group2[j,])^2))
            between_dist <- c(between_dist, d)
          }
        }
      }
    }
  }

  list(
    within_mean = mean(within_dist),
    within_sd = sd(within_dist),
    between_mean = mean(between_dist),
    between_sd = sd(between_dist),
    separation_ratio = mean(between_dist) / mean(within_dist)
  )
}

cat("\n=== Clustering Quality Metrics ===\n\n")

rust_metrics <- calc_separation(rust_df)
cat("Rust Implementation:\n")
cat(sprintf("  Within-group distance: %.3f (±%.3f)\n",
            rust_metrics$within_mean, rust_metrics$within_sd))
cat(sprintf("  Between-group distance: %.3f (±%.3f)\n",
            rust_metrics$between_mean, rust_metrics$between_sd))
cat(sprintf("  Separation ratio: %.3f\n\n", rust_metrics$separation_ratio))

uwot_metrics <- calc_separation(uwot_df)
cat("R uwot Implementation:\n")
cat(sprintf("  Within-group distance: %.3f (±%.3f)\n",
            uwot_metrics$within_mean, uwot_metrics$within_sd))
cat(sprintf("  Between-group distance: %.3f (±%.3f)\n",
            uwot_metrics$between_mean, uwot_metrics$between_sd))
cat(sprintf("  Separation ratio: %.3f\n", uwot_metrics$separation_ratio))
