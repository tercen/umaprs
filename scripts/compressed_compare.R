#!/usr/bin/env Rscript
library(uwot)
library(ggplot2)
library(gridExtra)
library(cowplot)

data <- as.matrix(read.csv("data/cyto_data_clean.csv"))
labels <- read.csv("data/cyto_labels.csv")
n <- nrow(data)

set.seed(42)
t0 <- proc.time()
emb <- umap(data, n_neighbors=15, n_components=2, min_dist=0.1, n_epochs=200, verbose=FALSE)
t_uwot <- (proc.time() - t0)["elapsed"]
uwot_emb <- data.frame(V1=emb[,1], V2=emb[,2])

read_if <- function(p) if (file.exists(p)) read.csv(p) else NULL
embs <- list(
    "Standard"      = read_if("results/cyto_emb_standard.csv"),
    "Compressed TQ4" = read_if("results/cyto_emb_compressed_tq4.csv"),
    "Compressed TQ8" = read_if("results/cyto_emb_compressed_tq8.csv"),
    "R uwot"        = uwot_emb
)
times <- list("Standard"=6.5, "Compressed TQ4"=6.6, "Compressed TQ8"=6.8, "R uwot"=t_uwot)
mem <- list("Standard"="12.2 MB", "Compressed TQ4"="1.0 MB", "Compressed TQ8"="1.7 MB", "R uwot"="12.2 MB")

calc_sep <- function(emb) {
    set.seed(1); w <- c(); b <- c()
    for (p in 1:80000) { i <- sample(n,1); j <- sample(n,1); if(i==j) next
        d <- sqrt((emb$V1[i]-emb$V1[j])^2+(emb$V2[i]-emb$V2[j])^2)
        if(labels$population[i]==labels$population[j]) w<-c(w,d) else b<-c(b,d) }
    mean(b)/mean(w)
}

seps <- list()
for (name in names(embs)) seps[[name]] <- calc_sep(embs[[name]])
uwot_sep <- seps[["R uwot"]]

cat(sprintf("\n%-18s %8s %6s %8s %10s\n", "Method", "Sep", "vs uwot", "Time", "Memory"))
for (name in names(embs)) {
    cat(sprintf("%-18s %8.3f %5.0f%% %7.1fs %10s\n",
        name, seps[[name]], 100*seps[[name]]/uwot_sep, times[[name]], mem[[name]]))
}

set.seed(1); idx <- sample(n, min(20000, n))
pop_colors <- c("1"="#e6194b","2"="#3cb44b","3"="#4363d8","4"="#f58231","5"="#911eb4",
"6"="#42d4f4","7"="#f032e6","8"="#bfef45","9"="#fabebe","10"="#469990",
"11"="#e6beff","12"="#9A6324","13"="#800000","14"="#000075")

plots <- list()
for (name in names(embs)) {
    emb <- embs[[name]]
    sep <- seps[[name]]
    t <- times[[name]]
    m <- mem[[name]]

    df <- data.frame(x=emb$V1[idx], y=emb$V2[idx], pop=factor(labels$population[idx]))
    df <- df[sample(nrow(df)),]

    p <- ggplot(df, aes(x=x,y=y,color=pop)) + geom_point(size=0.3,alpha=0.5) +
        scale_color_manual(values=pop_colors) +
        labs(title=name,
             subtitle=sprintf("%s | %.1fs | Sep: %.2f (%+.0f%%)", m, t, sep, 100*sep/uwot_sep - 100),
             x="UMAP 1", y="UMAP 2") +
        theme_minimal() + theme(legend.position="none",
            plot.title=element_text(hjust=0.5, face="bold", size=11),
            plot.subtitle=element_text(hjust=0.5, size=7))
    plots[[length(plots)+1]] <- p
}

df_leg <- data.frame(x=0, y=0, pop=factor(1:14))
p_leg <- ggplot(df_leg, aes(x=x,y=y,color=pop)) + geom_point(size=3) +
    scale_color_manual(values=pop_colors, name="Population") +
    theme_void() + theme(legend.position="bottom") +
    guides(color=guide_legend(nrow=1, override.aes=list(size=3, alpha=1)))
legend <- get_legend(p_leg)

png("results/compressed_compare.png", width=1800, height=700, res=120)
grid.arrange(
    do.call(arrangeGrob, c(plots, list(ncol=4))),
    legend, nrow=2, heights=c(5,1)
)
dev.off()
cat("\nSaved: results/compressed_compare.png\n")
