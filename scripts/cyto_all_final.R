#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)
library(cowplot)

data <- as.matrix(read.csv("data/cyto_data_clean.csv"))
labels <- read.csv("data/cyto_labels.csv")
n <- nrow(data)
cat(sprintf("CyTOF: %d cells x %d markers, %d populations\n\n", n, ncol(data), length(unique(labels$population))))

set.seed(42)
t0 <- proc.time()
emb <- umap(data, n_neighbors=15, n_components=2, min_dist=0.1, n_epochs=200, verbose=FALSE)
t_uwot <- (proc.time() - t0)["elapsed"]
uwot_emb <- data.frame(V1=emb[,1], V2=emb[,2])

timings <- read.csv("results/timings.csv")
read_if <- function(path) if (file.exists(path)) read.csv(path) else NULL

# Ordered: kd-tree, TQ4 CPU, TQ4 GPU, TQ8 CPU, TQ8 GPU, train 10%, uwot
ordered_methods <- c("kd-tree", "TQ4+QJL", "GPU TQ4", "TQ8+QJL", "GPU TQ8", "train 10%", "R uwot")

embs <- list(
    "kd-tree"    = read_if("results/cyto_emb_kdtree.csv"),
    "TQ4+QJL"   = read_if("results/cyto_emb_tq4.csv"),
    "GPU TQ4"    = read_if("results/cyto_emb_gpu_tq4.csv"),
    "TQ8+QJL"   = read_if("results/cyto_emb_tq8.csv"),
    "GPU TQ8"    = read_if("results/cyto_emb_gpu_tq8.csv"),
    "train 10%"  = read_if("results/cyto_emb_train10.csv"),
    "R uwot"     = uwot_emb
)

calc_sep <- function(emb) {
    set.seed(1); w <- c(); b <- c()
    for (p in 1:80000) { i <- sample(n,1); j <- sample(n,1); if(i==j) next
        d <- sqrt((emb$V1[i]-emb$V1[j])^2+(emb$V2[i]-emb$V2[j])^2)
        if(labels$population[i]==labels$population[j]) w<-c(w,d) else b<-c(b,d) }
    mean(b)/mean(w)
}

results <- data.frame(method=character(), sep=numeric(), time=numeric(), stringsAsFactors=FALSE)
for (name in ordered_methods) {
    if (is.null(embs[[name]])) next
    sep <- calc_sep(embs[[name]])
    t <- if (name == "R uwot") t_uwot else {
        row <- timings[timings$method == name, ]
        if (nrow(row) > 0) row$time[1] else NA
    }
    results <- rbind(results, data.frame(method=name, sep=sep, time=t))
}

uwot_sep <- results$sep[results$method == "R uwot"]
cat(sprintf("\n%-14s %8s %6s %8s\n", "Method", "Sep", "vs uwot", "Time"))
for (i in 1:nrow(results)) {
    cat(sprintf("%-14s %8.3f %5.0f%% %7.1fs\n",
        results$method[i], results$sep[i], 100*results$sep[i]/uwot_sep, results$time[i]))
}

# Plot
set.seed(1); idx <- sample(n, min(20000, n))
pop_colors <- c("1"="#e6194b","2"="#3cb44b","3"="#4363d8","4"="#f58231","5"="#911eb4",
"6"="#42d4f4","7"="#f032e6","8"="#bfef45","9"="#fabebe","10"="#469990",
"11"="#e6beff","12"="#9A6324","13"="#800000","14"="#000075")

plots <- list()
for (i in 1:nrow(results)) {
    name <- results$method[i]
    emb <- embs[[name]]
    sep <- results$sep[i]
    time <- results$time[i]

    df <- data.frame(x=emb$V1[idx], y=emb$V2[idx], pop=factor(labels$population[idx]))
    df <- df[sample(nrow(df)),]

    p <- ggplot(df, aes(x=x,y=y,color=pop)) + geom_point(size=0.3,alpha=0.5) +
        scale_color_manual(values=pop_colors) +
        labs(title=name,
             subtitle=sprintf("%.1fs | Sep: %.2f (%+.0f%%)", time, sep, 100*sep/uwot_sep - 100),
             x="UMAP 1", y="UMAP 2") +
        theme_minimal() + theme(legend.position="none",
            plot.title=element_text(hjust=0.5, face="bold", size=10),
            plot.subtitle=element_text(hjust=0.5, size=7))
    plots[[length(plots)+1]] <- p
}

df_leg <- data.frame(x=0, y=0, pop=factor(1:14))
p_leg <- ggplot(df_leg, aes(x=x,y=y,color=pop)) + geom_point(size=3) +
    scale_color_manual(values=pop_colors, name="Population") +
    theme_void() + theme(legend.position="bottom") +
    guides(color=guide_legend(nrow=1, override.aes=list(size=3, alpha=1)))
legend <- get_legend(p_leg)

ncols <- length(plots)
png("results/cyto_all_final.png", width=ncols*420, height=700, res=120)
grid.arrange(
    do.call(arrangeGrob, c(plots, list(ncol=ncols))),
    legend, nrow=2, heights=c(5,1)
)
dev.off()
cat(sprintf("\nSaved: results/cyto_all_final.png (%d panels)\n", ncols))
