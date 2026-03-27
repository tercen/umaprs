#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)
library(cowplot)

labels <- read.csv("data/cyto_labels.csv")
n <- nrow(labels)

# Read all embeddings
read_if <- function(p) if (file.exists(p)) read.csv(p) else NULL

embs <- list(
    "Standard\n(12.2 MB)"       = read_if("results/cyto_emb_standard.csv"),
    "Std 14%\n(~1.7 MB)"        = read_if("results/budget_std_7x.csv"),
    "TQ8 compressed\n(1.7 MB)"  = read_if("results/budget_tq8.csv"),
    "Std 8%\n(~1.0 MB)"         = read_if("results/budget_std_12x.csv"),
    "TQ4 compressed\n(1.0 MB)"  = read_if("results/budget_tq4.csv")
)

calc_sep <- function(emb) {
    set.seed(1); w <- c(); b <- c()
    for (p in 1:80000) { i <- sample(n,1); j <- sample(n,1); if(i==j) next
        d <- sqrt((emb$V1[i]-emb$V1[j])^2+(emb$V2[i]-emb$V2[j])^2)
        if(labels$population[i]==labels$population[j]) w<-c(w,d) else b<-c(b,d) }
    mean(b)/mean(w)
}

seps <- list()
for (name in names(embs)) {
    if (!is.null(embs[[name]])) seps[[name]] <- calc_sep(embs[[name]])
}
ref_sep <- seps[["Standard\n(12.2 MB)"]]

cat(sprintf("\n%-25s %8s %6s\n", "Method", "Sep", "vs exact"))
for (name in names(seps)) {
    cat(sprintf("%-25s %8.3f %5.0f%%\n", gsub("\n", " ", name), seps[[name]], 100*seps[[name]]/ref_sep))
}

set.seed(1); idx <- sample(n, min(20000, n))
pop_colors <- c("1"="#e6194b","2"="#3cb44b","3"="#4363d8","4"="#f58231","5"="#911eb4",
"6"="#42d4f4","7"="#f032e6","8"="#bfef45","9"="#fabebe","10"="#469990",
"11"="#e6beff","12"="#9A6324","13"="#800000","14"="#000075")

plots <- list()
for (name in names(embs)) {
    emb <- embs[[name]]
    if (is.null(emb)) next
    sep <- seps[[name]]

    df <- data.frame(x=emb$V1[idx], y=emb$V2[idx], pop=factor(labels$population[idx]))
    df <- df[sample(nrow(df)),]

    p <- ggplot(df, aes(x=x,y=y,color=pop)) + geom_point(size=0.3,alpha=0.5) +
        scale_color_manual(values=pop_colors) +
        labs(title=name,
             subtitle=sprintf("Sep: %.2f (%+.0f%%)", sep, 100*sep/ref_sep - 100),
             x="UMAP 1", y="UMAP 2") +
        theme_minimal() + theme(legend.position="none",
            plot.title=element_text(hjust=0.5, face="bold", size=10),
            plot.subtitle=element_text(hjust=0.5, size=8))
    plots[[length(plots)+1]] <- p
}

df_leg <- data.frame(x=0, y=0, pop=factor(1:14))
p_leg <- ggplot(df_leg, aes(x=x,y=y,color=pop)) + geom_point(size=3) +
    scale_color_manual(values=pop_colors, name="Population") +
    theme_void() + theme(legend.position="bottom") +
    guides(color=guide_legend(nrow=1, override.aes=list(size=3, alpha=1)))
legend <- get_legend(p_leg)

ncols <- length(plots)
png("results/memory_budget.png", width=ncols*420, height=700, res=120)
grid.arrange(
    do.call(arrangeGrob, c(plots, list(ncol=ncols))),
    legend, nrow=2, heights=c(5,1)
)
dev.off()
cat(sprintf("\nSaved: results/memory_budget.png (%d panels)\n", ncols))
