#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)

data <- as.matrix(read.csv("highd_data.csv", header = FALSE))
labels <- read.csv("highd_labels.csv")

# Run uwot
set.seed(42)
uwot_emb <- umap(data, n_neighbors = 15, n_components = 2, min_dist = 0.1,
                  n_epochs = 200, verbose = FALSE)
uwot_emb <- data.frame(V1 = uwot_emb[, 1], V2 = uwot_emb[, 2])

std_emb <- read.csv("highd_emb_standard.csv")
tq4_emb <- read.csv("highd_emb_tq4.csv")
tq8_emb <- read.csv("highd_emb_tq8.csv")

calc_sep <- function(emb, labs) {
  n <- nrow(emb)
  set.seed(1)
  within_d <- c(); between_d <- c()
  for (p in 1:min(40000, n*(n-1)/2)) {
    i <- sample(n, 1); j <- sample(n, 1)
    if (i == j) next
    d <- sqrt((emb$V1[i] - emb$V1[j])^2 + (emb$V2[i] - emb$V2[j])^2)
    if (labs$cluster[i] == labs$cluster[j]) {
      within_d <- c(within_d, d)
    } else {
      between_d <- c(between_d, d)
    }
  }
  mean(between_d) / mean(within_d)
}

sep_std  <- calc_sep(std_emb, labels)
sep_tq4  <- calc_sep(tq4_emb, labels)
sep_tq8  <- calc_sep(tq8_emb, labels)
sep_uwot <- calc_sep(uwot_emb, labels)

cat(sprintf("\n=== 500 points x 200 dims, 4 clusters ===\n\n"))
cat(sprintf("Separation ratios:\n"))
cat(sprintf("  Rust Standard (exact kNN):   %.3f  (%.1f%%)\n", sep_std, 100*sep_std/sep_uwot))
cat(sprintf("  Rust TurboQuant 4-bit+HNSW:  %.3f  (%.1f%%)\n", sep_tq4, 100*sep_tq4/sep_uwot))
cat(sprintf("  Rust TurboQuant 8-bit+HNSW:  %.3f  (%.1f%%)\n", sep_tq8, 100*sep_tq8/sep_uwot))
cat(sprintf("  R uwot (reference):          %.3f  (100%%)\n", sep_uwot))

make_plot <- function(emb, title, subtitle) {
  df <- data.frame(x = emb$V1, y = emb$V2, cluster = factor(labels$cluster))
  ggplot(df, aes(x = x, y = y, color = cluster)) +
    geom_point(size = 1.8, alpha = 0.7) +
    scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#CC79A7")) +
    labs(title = title, subtitle = subtitle, x = "UMAP 1", y = "UMAP 2") +
    theme_minimal() +
    theme(legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
          plot.subtitle = element_text(hjust = 0.5, size = 8))
}

p1 <- make_plot(std_emb, "Rust Standard",
                sprintf("Exact kNN | Sep: %.3f", sep_std))
p2 <- make_plot(tq4_emb, "Rust TQ 4-bit",
                sprintf("4-bit HNSW (12x compr.) | Sep: %.3f", sep_tq4))
p3 <- make_plot(tq8_emb, "Rust TQ 8-bit",
                sprintf("8-bit HNSW (6x compr.) | Sep: %.3f", sep_tq8))
p4 <- make_plot(uwot_emb, "R uwot",
                sprintf("Reference | Sep: %.3f", sep_uwot))

png("highd_compare.png", width = 2000, height = 550, res = 120)
grid.arrange(p1, p2, p3, p4, ncol = 4)
dev.off()

cat("\nSaved: highd_compare.png\n")
