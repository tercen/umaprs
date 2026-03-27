#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)

data <- as.matrix(read.csv("data/cyto_data_clean.csv"))
labels <- read.csv("data/cyto_labels.csv")
cat(sprintf("CyTOF: %d cells x %d markers, %d populations\n\n",
            nrow(data), ncol(data), length(unique(labels$population))))

# Exact same params
for (run in 1:3) {
    set.seed(42)
    t0 <- proc.time()
    emb <- umap(data,
                n_neighbors = 15,
                n_components = 2,
                min_dist = 0.1,
                spread = 1.0,
                learning_rate = 1.0,
                n_epochs = 200,
                negative_sample_rate = 5,
                repulsion_strength = 1.0,
                verbose = FALSE)
    elapsed <- (proc.time() - t0)["elapsed"]
    cat(sprintf("uwot run %d: %.3fs\n", run, elapsed))
}

uwot_emb <- data.frame(V1 = emb[, 1], V2 = emb[, 2])
rust_emb <- read.csv("cyto_emb_rust.csv")

# Separation metric
calc_sep <- function(emb, labs) {
    n <- nrow(emb); set.seed(1)
    w <- c(); b <- c()
    for (p in 1:80000) {
        i <- sample(n, 1); j <- sample(n, 1)
        if (i == j) next
        d <- sqrt((emb$V1[i] - emb$V1[j])^2 + (emb$V2[i] - emb$V2[j])^2)
        if (labs$population[i] == labs$population[j]) w <- c(w, d) else b <- c(b, d)
    }
    mean(b) / mean(w)
}

sep_rust <- calc_sep(rust_emb, labels)
sep_uwot <- calc_sep(uwot_emb, labels)

cat(sprintf("\nSeparation:\n  Rust: %.3f  (%.1f%%)\n  uwot: %.3f  (100%%)\n",
            sep_rust, 100 * sep_rust / sep_uwot, sep_uwot))

# Plot
set.seed(1)
n_plot <- min(20000, nrow(rust_emb))
idx <- sample(nrow(rust_emb), n_plot)

pop_colors <- c(
    "1"="#e6194b", "2"="#3cb44b", "3"="#4363d8", "4"="#f58231", "5"="#911eb4",
    "6"="#42d4f4", "7"="#f032e6", "8"="#bfef45", "9"="#fabebe", "10"="#469990",
    "11"="#e6beff", "12"="#9A6324", "13"="#800000", "14"="#000075"
)

make_plot <- function(emb, title, subtitle) {
    df <- data.frame(x = emb$V1[idx], y = emb$V2[idx],
                     pop = factor(labels$population[idx]))
    df <- df[sample(nrow(df)), ]
    ggplot(df, aes(x = x, y = y, color = pop)) +
        geom_point(size = 0.4, alpha = 0.5) +
        scale_color_manual(values = pop_colors) +
        labs(title = title, subtitle = subtitle, x = "UMAP 1", y = "UMAP 2") +
        theme_minimal() +
        theme(legend.position = "right",
              legend.key.size = unit(0.4, "cm"),
              plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
              plot.subtitle = element_text(hjust = 0.5, size = 9)) +
        guides(color = guide_legend(override.aes = list(size = 3, alpha = 1)))
}

p1 <- make_plot(rust_emb, "Rust UMAP",
                sprintf("Sep: %.3f", sep_rust))
p2 <- make_plot(uwot_emb, "R uwot",
                sprintf("Sep: %.3f", sep_uwot))

png("cyto_final.png", width = 1600, height = 700, res = 120)
grid.arrange(p1, p2, ncol = 2)
dev.off()
cat("Saved: cyto_final.png\n")
