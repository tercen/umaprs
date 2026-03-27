#!/usr/bin/env Rscript

library(ggplot2)
library(gridExtra)

labels <- read.csv("data/crabs_labels.csv")
rust_embedding <- read.csv("rust_crabs_embedding.csv")
uwot_embedding <- read.csv("uwot_crabs_embedding.csv")

calc_stats <- function(embedding, name) {
  cat(name, ":\n", sep = "")
  cat("  Std X:", sprintf("%.3f", sd(embedding$V1)), "\n")
  cat("  Std Y:", sprintf("%.3f", sd(embedding$V2)), "\n")

  embedding$group <- labels$group
  within_dist <- c()
  between_dist <- c()
  groups <- unique(labels$group)

  for (g1 in groups) {
    points_g1 <- embedding[embedding$group == g1, c("V1", "V2")]
    if (nrow(points_g1) > 1) {
      within_dist <- c(within_dist, as.vector(dist(points_g1)))
    }
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

  sep <- mean(between_dist) / mean(within_dist)
  cat("  Within-group dist:", sprintf("%.3f", mean(within_dist)), "\n")
  cat("  Between-group dist:", sprintf("%.3f", mean(between_dist)), "\n")
  cat("  Separation ratio:", sprintf("%.3f", sep), "\n\n")
  return(list(sep = sep, within = mean(within_dist), between = mean(between_dist)))
}

cat("=== Rust vs uwot Comparison ===\n\n")
rust_stats <- calc_stats(rust_embedding, "Rust")
uwot_stats <- calc_stats(uwot_embedding, "uwot")

cat("Quality (sep ratio): ", sprintf("%.1f%%", 100 * rust_stats$sep / uwot_stats$sep), "\n\n")

create_plot <- function(embedding, title) {
  df <- data.frame(x = embedding$V1, y = embedding$V2, group = labels$group)
  ggplot(df, aes(x = x, y = y, color = group, shape = group)) +
    geom_point(size = 2.5, alpha = 0.7) +
    scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442")) +
    scale_shape_manual(values = c(16, 17, 15, 18)) +
    labs(title = title, x = "UMAP 1", y = "UMAP 2") +
    theme_minimal() +
    theme(legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, face = "bold", size = 14)) +
    coord_fixed()
}

p1 <- create_plot(rust_embedding, "Rust UMAP")
p2 <- create_plot(uwot_embedding, "R uwot UMAP")

png("comparison_crabs.png", width = 1200, height = 600, res = 100)
grid.arrange(p1, p2, ncol = 2)
dev.off()

cat("Saved: comparison_crabs.png\n")
