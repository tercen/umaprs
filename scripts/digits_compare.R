#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)

data <- as.matrix(read.csv("data/digits_data.csv"))
labels <- read.csv("data/digits_labels.csv")

cat("Optical Digits:", nrow(data), "samples,", ncol(data), "features, 10 classes\n\n")

# Run uwot
set.seed(42)
t0 <- proc.time()
uwot_emb <- umap(data, n_neighbors = 15, n_components = 2, min_dist = 0.1,
                  n_epochs = 200, verbose = FALSE)
t_uwot <- (proc.time() - t0)["elapsed"]
uwot_emb <- data.frame(V1 = uwot_emb[, 1], V2 = uwot_emb[, 2])

tq4_emb <- read.csv("digits_emb_tq4.csv")
tq8_emb <- read.csv("digits_emb_tq8.csv")

# Separation metric (sampled)
calc_sep <- function(emb, labs) {
  n <- nrow(emb)
  set.seed(1)
  within_d <- c(); between_d <- c()
  for (p in 1:50000) {
    i <- sample(n, 1); j <- sample(n, 1)
    if (i == j) next
    d <- sqrt((emb$V1[i] - emb$V1[j])^2 + (emb$V2[i] - emb$V2[j])^2)
    if (labs$digit[i] == labs$digit[j]) {
      within_d <- c(within_d, d)
    } else {
      between_d <- c(between_d, d)
    }
  }
  mean(between_d) / mean(within_d)
}

sep_tq4  <- calc_sep(tq4_emb, labels)
sep_tq8  <- calc_sep(tq8_emb, labels)
sep_uwot <- calc_sep(uwot_emb, labels)

cat(sprintf("Separation ratios:\n"))
cat(sprintf("  Rust TQ 4-bit + HNSW:  %.3f  (%.1f%%)\n", sep_tq4, 100*sep_tq4/sep_uwot))
cat(sprintf("  Rust TQ 8-bit + HNSW:  %.3f  (%.1f%%)\n", sep_tq8, 100*sep_tq8/sep_uwot))
cat(sprintf("  R uwot:                %.3f  (100%%)\n", sep_uwot))
cat(sprintf("\nuwot time: %.2fs\n", t_uwot))

# Colors for 10 digits
digit_colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

make_plot <- function(emb, title, subtitle) {
  df <- data.frame(x = emb$V1, y = emb$V2, digit = factor(labels$digit))
  ggplot(df, aes(x = x, y = y, color = digit)) +
    geom_point(size = 0.6, alpha = 0.5) +
    scale_color_manual(values = digit_colors) +
    labs(title = title, subtitle = subtitle, x = "UMAP 1", y = "UMAP 2") +
    theme_minimal() +
    theme(legend.position = "right",
          legend.key.size = unit(0.3, "cm"),
          plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
          plot.subtitle = element_text(hjust = 0.5, size = 8)) +
    guides(color = guide_legend(override.aes = list(size = 2, alpha = 1)))
}

p1 <- make_plot(tq4_emb, "Rust TQ 4-bit",
                sprintf("12x compression | Sep: %.3f", sep_tq4))
p2 <- make_plot(tq8_emb, "Rust TQ 8-bit",
                sprintf("6x compression | Sep: %.3f", sep_tq8))
p3 <- make_plot(uwot_emb, "R uwot",
                sprintf("Reference | Sep: %.3f", sep_uwot))

png("digits_compare.png", width = 1800, height = 550, res = 120)
grid.arrange(p1, p2, p3, ncol = 3)
dev.off()

cat("\nSaved: digits_compare.png\n")
