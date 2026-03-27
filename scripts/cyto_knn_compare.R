#!/usr/bin/env Rscript
library(ggplot2)
library(gridExtra)

labels <- read.csv("data/cyto_labels.csv")

calc_sep <- function(emb, labs, n) {
    set.seed(1); w <- c(); b <- c()
    for (p in 1:50000) {
        i <- sample(n, 1); j <- sample(n, 1)
        if (i == j) next
        d <- sqrt((emb$V1[i] - emb$V1[j])^2 + (emb$V2[i] - emb$V2[j])^2)
        if (labs$population[i] == labs$population[j]) w <- c(w, d) else b <- c(b, d)
    }
    mean(b) / mean(w)
}

tq4 <- read.csv("cyto_emb_tq4.csv")
tq8 <- read.csv("cyto_emb_tq8.csv")
exact10k <- read.csv("cyto_emb_exact10k.csv")
tq4_10k <- read.csv("cyto_emb_tq4_10k.csv")

s1 <- calc_sep(tq4, labels, 50000)
s2 <- calc_sep(tq8, labels, 50000)
s3 <- calc_sep(exact10k, labels, 10000)
s4 <- calc_sep(tq4_10k, labels, 10000)

cat(sprintf("50k: TQ 4-bit sep = %.3f\n", s1))
cat(sprintf("50k: TQ 8-bit sep = %.3f\n", s2))
cat(sprintf("10k: Exact    sep = %.3f\n", s3))
cat(sprintf("10k: TQ 4-bit sep = %.3f\n", s4))
