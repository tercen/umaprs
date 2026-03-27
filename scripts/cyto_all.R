#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)
library(cowplot)

data <- as.matrix(read.csv("data/cyto_data_clean.csv"))
labels <- read.csv("data/cyto_labels.csv")
n <- nrow(data)

# uwot
set.seed(42)
t0 <- proc.time()
emb <- umap(data, n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0,
            learning_rate=1.0, n_epochs=200, negative_sample_rate=5,
            repulsion_strength=1.0, verbose=FALSE)
t_uwot <- (proc.time() - t0)["elapsed"]
uwot_emb <- data.frame(V1=emb[,1], V2=emb[,2])

# Read all Rust embeddings
kd_emb     <- read.csv("cyto_emb_kdtree.csv")
tq4h_emb   <- read.csv("cyto_emb_tq4.csv")
tq8h_emb   <- read.csv("cyto_emb_tq8_hnsw.csv")
tq4kd_emb  <- read.csv("cyto_emb_tq4_kd.csv")
tq8kd_emb  <- read.csv("cyto_emb_tq8_kd.csv")

calc_sep <- function(emb) {
    set.seed(1); w <- c(); b <- c()
    for (p in 1:80000) { i <- sample(n,1); j <- sample(n,1); if(i==j) next
        d <- sqrt((emb$V1[i]-emb$V1[j])^2+(emb$V2[i]-emb$V2[j])^2)
        if(labels$population[i]==labels$population[j]) w<-c(w,d) else b<-c(b,d) }
    mean(b)/mean(w)
}

s <- list(
    kd     = calc_sep(kd_emb),
    tq4h   = calc_sep(tq4h_emb),
    tq8h   = calc_sep(tq8h_emb),
    tq4kd  = calc_sep(tq4kd_emb),
    tq8kd  = calc_sep(tq8kd_emb),
    uwot   = calc_sep(uwot_emb)
)

cat(sprintf("\n%-20s %8s %6s\n", "Method", "Sep", "vs uwot"))
cat(sprintf("%-20s %8.3f %5.0f%%\n", "kd-tree (exact)", s$kd, 100*s$kd/s$uwot))
cat(sprintf("%-20s %8.3f %5.0f%%\n", "TQ4 + HNSW", s$tq4h, 100*s$tq4h/s$uwot))
cat(sprintf("%-20s %8.3f %5.0f%%\n", "TQ8 + HNSW", s$tq8h, 100*s$tq8h/s$uwot))
cat(sprintf("%-20s %8.3f %5.0f%%\n", "TQ4 + kd-tree", s$tq4kd, 100*s$tq4kd/s$uwot))
cat(sprintf("%-20s %8.3f %5.0f%%\n", "TQ8 + kd-tree", s$tq8kd, 100*s$tq8kd/s$uwot))
cat(sprintf("%-20s %8.3f %5.0f%%\n", "uwot (FNN)", s$uwot, 100))

# Plot all 6
set.seed(1)
idx <- sample(n, min(20000, n))

pop_colors <- c(
    "1"="#e6194b", "2"="#3cb44b", "3"="#4363d8", "4"="#f58231", "5"="#911eb4",
    "6"="#42d4f4", "7"="#f032e6", "8"="#bfef45", "9"="#fabebe", "10"="#469990",
    "11"="#e6beff", "12"="#9A6324", "13"="#800000", "14"="#000075"
)

make_plot <- function(emb, title, sub) {
    df <- data.frame(x=emb$V1[idx], y=emb$V2[idx], pop=factor(labels$population[idx]))
    df <- df[sample(nrow(df)), ]
    ggplot(df, aes(x=x, y=y, color=pop)) +
        geom_point(size=0.3, alpha=0.5) +
        scale_color_manual(values=pop_colors) +
        labs(title=title, subtitle=sub, x="UMAP 1", y="UMAP 2") +
        theme_minimal() +
        theme(legend.position="none",
              plot.title=element_text(hjust=0.5, face="bold", size=10),
              plot.subtitle=element_text(hjust=0.5, size=7))
}

p1 <- make_plot(kd_emb,    "kd-tree",      sprintf("Sep: %.2f", s$kd))
p2 <- make_plot(tq4h_emb,  "TQ4+HNSW",     sprintf("Sep: %.2f", s$tq4h))
p3 <- make_plot(tq8h_emb,  "TQ8+HNSW",     sprintf("Sep: %.2f", s$tq8h))
p4 <- make_plot(tq4kd_emb, "TQ4+kd-tree",  sprintf("Sep: %.2f", s$tq4kd))
p5 <- make_plot(tq8kd_emb, "TQ8+kd-tree",  sprintf("Sep: %.2f", s$tq8kd))
p6 <- make_plot(uwot_emb,  "uwot",         sprintf("Sep: %.2f", s$uwot))

df_leg <- data.frame(x=0, y=0, pop=factor(1:14))
p_leg <- ggplot(df_leg, aes(x=x,y=y,color=pop)) + geom_point(size=3) +
    scale_color_manual(values=pop_colors, name="Population") +
    theme_void() + theme(legend.position="bottom") +
    guides(color=guide_legend(nrow=1, override.aes=list(size=3, alpha=1)))
legend <- get_legend(p_leg)

png("cyto_all.png", width=2400, height=700, res=120)
grid.arrange(
    arrangeGrob(p1, p2, p3, p4, p5, p6, ncol=6),
    legend, nrow=2, heights=c(5,1)
)
dev.off()
cat("Saved: cyto_all.png\n")
