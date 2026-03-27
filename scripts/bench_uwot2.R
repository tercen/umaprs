#!/usr/bin/env Rscript
library(uwot)

configs <- list(
  c(1000, 50), c(1000, 200), c(1000, 500),
  c(5000, 50), c(5000, 200),
  c(10000, 50), c(10000, 200)
)

for (cfg in configs) {
  n <- cfg[1]; d <- cfg[2]
  data <- matrix(0, nrow = n, ncol = d)
  for (i in 1:n) {
    for (j in 1:d) {
      data[i, j] <- (i - 1) * (j - 1) / n + sin((i - 1) * 0.1)
    }
  }

  start <- proc.time()
  embedding <- umap(data, n_neighbors = 15, n_components = 2, min_dist = 0.1,
                    n_epochs = 200, verbose = FALSE)
  elapsed <- (proc.time() - start)["elapsed"]

  cat(sprintf("n=%5d, dims=%3d, time=%.3fs\n", n, d, elapsed))
}
