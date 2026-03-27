#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)

labels <- read.csv("data/crabs_labels.csv")

# Run uwot
set.seed(42)
data <- read.csv("data/crabs_data.csv")
uwot_emb <- umap(data, n_neighbors = 15, n_components = 2, min_dist = 0.1,
                  n_epochs = 200, verbose = FALSE)
uwot_emb <- data.frame(V1 = uwot_emb[, 1], V2 = uwot_emb[, 2])

# Read Rust embeddings
standard_emb <- read.csv("crabs_emb_standard.csv")
tq4_emb <- read.csv("crabs_emb_tq4.csv")
tq8_emb <- read.csv("crabs_emb_tq8.csv")

# Compute separation
calc_sep <- function(emb, labs) {
  emb$group <- labs$group
  within_d <- c(); between_d <- c()
  groups <- unique(labs$group)
  for (g1 in groups) {
    p1 <- emb[emb$group == g1, c("V1", "V2")]
    if (nrow(p1) > 1) within_d <- c(within_d, as.vector(dist(p1)))
    for (g2 in groups) {
      if (g1 < g2) {
        p2 <- emb[emb$group == g2, c("V1", "V2")]
        for (i in 1:nrow(p1)) for (j in 1:nrow(p2))
          between_d <- c(between_d, sqrt((p1[i,1]-p2[j,1])^2 + (p1[i,2]-p2[j,2])^2))
      }
    }
  }
  mean(between_d) / mean(within_d)
}

sep_std  <- calc_sep(standard_emb, labels)
sep_tq4  <- calc_sep(tq4_emb, labels)
sep_tq8  <- calc_sep(tq8_emb, labels)
sep_uwot <- calc_sep(uwot_emb, labels)

cat(sprintf("Separation ratios:\n"))
cat(sprintf("  Rust Standard (exact kNN):   %.3f  (%.1f%%)\n", sep_std, 100*sep_std/sep_uwot))
cat(sprintf("  Rust TurboQuant 4-bit+HNSW:  %.3f  (%.1f%%)\n", sep_tq4, 100*sep_tq4/sep_uwot))
cat(sprintf("  Rust TurboQuant 8-bit+HNSW:  %.3f  (%.1f%%)\n", sep_tq8, 100*sep_tq8/sep_uwot))
cat(sprintf("  R uwot (reference):          %.3f  (100%%)\n", sep_uwot))

# Plot
make_plot <- function(emb, title, subtitle) {
  df <- data.frame(x = emb$V1, y = emb$V2, group = labels$group)
  ggplot(df, aes(x = x, y = y, color = group, shape = group)) +
    geom_point(size = 2.5, alpha = 0.7) +
    scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442")) +
    scale_shape_manual(values = c(16, 17, 15, 18)) +
    labs(title = title, subtitle = subtitle, x = "UMAP 1", y = "UMAP 2") +
    theme_minimal() +
    theme(legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
          plot.subtitle = element_text(hjust = 0.5, size = 8))
}

p1 <- make_plot(standard_emb, "Rust Standard",
                sprintf("Exact kNN | Sep: %.3f", sep_std))
p2 <- make_plot(tq4_emb, "Rust TQ 4-bit",
                sprintf("4-bit HNSW | Sep: %.3f", sep_tq4))
p3 <- make_plot(tq8_emb, "Rust TQ 8-bit",
                sprintf("8-bit HNSW | Sep: %.3f", sep_tq8))
p4 <- make_plot(uwot_emb, "R uwot",
                sprintf("Reference | Sep: %.3f", sep_uwot))

png("crabs_three_way.png", width = 2000, height = 550, res = 120)
grid.arrange(p1, p2, p3, p4, ncol = 4)
dev.off()

cat("\nSaved: crabs_three_way.png\n")
