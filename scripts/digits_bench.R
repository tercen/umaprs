#!/usr/bin/env Rscript
library(uwot)

data <- as.matrix(read.csv("data/digits_data.csv"))
cat(sprintf("Optical Digits: %d samples, %d features\n\n", nrow(data), ncol(data)))

for (i in 1:3) {
  set.seed(42)
  t0 <- proc.time()
  emb <- umap(data, n_neighbors = 15, n_components = 2, min_dist = 0.1,
              n_epochs = 200, verbose = FALSE)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("uwot run %d: %.3fs\n", i, elapsed))
}
