#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)

data <- as.matrix(read.csv("cyto_data.csv"))
labels <- read.csv("data/cyto_labels.csv")
cat(sprintf("Levine CyTOF: %d cells, %d markers, %d populations\n\n",
            nrow(data), ncol(data), length(unique(labels$population))))

# Run uwot
set.seed(42)
t0 <- proc.time()
emb <- umap(data, n_neighbors = 15, n_components = 2, min_dist = 0.1,
            n_epochs = 200, verbose = TRUE)
t_uwot <- (proc.time() - t0)["elapsed"]
cat(sprintf("\nuwot: %.3fs\n", t_uwot))

uwot_emb <- data.frame(V1 = emb[, 1], V2 = emb[, 2])
rust_emb <- read.csv("cyto_emb_rust.csv")

# Population distribution
cat("\nPopulation sizes:\n")
print(sort(table(labels$population), decreasing = TRUE))

# Better visualization: subsample for plotting, use better colors
set.seed(1)
n_plot <- min(20000, nrow(rust_emb))
idx <- sample(nrow(rust_emb), n_plot)

pop_colors <- c(
    "1"="#e6194b", "2"="#3cb44b", "3"="#4363d8", "4"="#f58231", "5"="#911eb4",
    "6"="#42d4f4", "7"="#f032e6", "8"="#bfef45", "9"="#fabebe", "10"="#469990",
    "11"="#e6beff", "12"="#9A6324", "13"="#800000", "14"="#000075"
)

make_plot <- function(emb, title) {
    df <- data.frame(x = emb$V1[idx], y = emb$V2[idx],
                     pop = factor(labels$population[idx]))
    # Shuffle to avoid overplotting bias
    df <- df[sample(nrow(df)), ]
    ggplot(df, aes(x = x, y = y, color = pop)) +
        geom_point(size = 0.4, alpha = 0.5) +
        scale_color_manual(values = pop_colors) +
        labs(title = title, x = "UMAP 1", y = "UMAP 2") +
        theme_minimal() +
        theme(legend.position = "right",
              legend.key.size = unit(0.4, "cm"),
              legend.text = element_text(size = 8),
              plot.title = element_text(hjust = 0.5, face = "bold", size = 13)) +
        guides(color = guide_legend(override.aes = list(size = 3, alpha = 1)))
}

p1 <- make_plot(rust_emb, "Rust UMAP (TurboQuant+HNSW)")
p2 <- make_plot(uwot_emb, "R uwot")

png("cyto_compare.png", width = 1600, height = 700, res = 120)
grid.arrange(p1, p2, ncol = 2)
dev.off()
cat("\nSaved: cyto_compare.png\n")
