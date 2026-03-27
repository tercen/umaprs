#!/usr/bin/env Rscript
library(uwot)

for (n in c(500, 1000, 2000, 5000, 10000)) {
  n_features <- 50
  data <- matrix(0, nrow = n, ncol = n_features)
  for (i in 1:n) {
    for (j in 1:n_features) {
      data[i, j] <- (i - 1) * (j - 1) / n + sin((i - 1) * 0.1)
    }
  }

  start <- proc.time()
  embedding <- umap(data, n_neighbors = 15, n_components = 2, min_dist = 0.1,
                    n_epochs = 200, verbose = FALSE)
  elapsed <- (proc.time() - start)["elapsed"]

  cat(sprintf("n=%5d, dims=%d, time=%.3fs, shape=[%d, %d]\n",
              n, n_features, elapsed, nrow(embedding), ncol(embedding)))
}
