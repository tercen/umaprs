#!/usr/bin/env Rscript
# uwot at three seeds for the envelope: Rscript scripts/envelope_uwot.R <name> <data.csv> [min_dist]
suppressMessages(library(uwot))
a <- commandArgs(trailingOnly = TRUE); name <- a[1]; md <- if (length(a) >= 3) as.numeric(a[3]) else 0.01
X <- as.matrix(read.csv(a[2])); secs <- c()
for (s in 1:3) {
  set.seed(s); t0 <- proc.time()[["elapsed"]]
  # the R umap_operator's call, at Jamie's settings; n_sgd_threads = 0 keeps the seed meaningful
  emb <- umap(X, n_neighbors = 15, min_dist = md, n_epochs = 200, init = "spectral", n_sgd_threads = 0)
  secs <- c(secs, proc.time()[["elapsed"]] - t0)
  write.table(emb, sprintf("results/env_%s_uwot_seed%d.csv", name, s), sep = ",", row.names = FALSE, col.names = FALSE)
}
writeLines(paste0("[", paste(sprintf("%.3f", secs), collapse = ","), "]"), sprintf("results/env_%s_uwot_timings.json", name))
cat("uwot done:", secs, "\n")
