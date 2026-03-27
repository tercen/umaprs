#!/usr/bin/env Rscript
library(ggplot2)
library(gridExtra)
library(cowplot)

for (pct in c("01pct", "1pct")) {
    labels <- read.csv(sprintf("data/aml_%s_labels.csv", pct))
    n <- nrow(labels)
    n_blast <- sum(labels$label == 1)
    pct_name <- if (pct == "01pct") "0.1%" else "1%"

    cat(sprintf("\n=== AML %s spike-in: %d cells, %d blasts ===\n", pct_name, n, n_blast))

    read_if <- function(p) if (file.exists(p)) read.csv(p) else NULL
    embs <- list(
        "Standard\n(exact, 12 MB)"   = read_if(sprintf("results/aml_%s_standard.csv", pct)),
        "Subsample 14%\n(~1.7 MB)"   = read_if(sprintf("results/aml_%s_sub14.csv", pct)),
        "TQ8 compressed\n(1.7 MB)"   = read_if(sprintf("results/aml_%s_tq8.csv", pct)),
        "Subsample 8%\n(~1.0 MB)"    = read_if(sprintf("results/aml_%s_sub8.csv", pct))
    )

    # For each method, measure: how many blast cells have a blast neighbor in embedding?
    # (proxy for "do blasts cluster together?")
    measure_blast_coherence <- function(emb, labs) {
        blast_idx <- which(labs$label == 1)
        if (length(blast_idx) < 2) return(NA)

        # For each blast cell, find its nearest neighbor in the embedding
        blast_coords <- emb[blast_idx, ]
        n_blast_with_blast_neighbor <- 0

        for (i in 1:length(blast_idx)) {
            dists <- sqrt((emb$V1 - blast_coords$V1[i])^2 + (emb$V2 - blast_coords$V2[i])^2)
            dists[blast_idx[i]] <- Inf  # exclude self
            nn <- which.min(dists)
            if (labs$label[nn] == 1) n_blast_with_blast_neighbor <- n_blast_with_blast_neighbor + 1
        }

        n_blast_with_blast_neighbor / length(blast_idx)
    }

    for (name in names(embs)) {
        emb <- embs[[name]]
        if (is.null(emb)) next
        coh <- measure_blast_coherence(emb, labels)
        cat(sprintf("  %-28s blast coherence: %.0f%% (%d/%d blasts have blast NN)\n",
                    gsub("\n", " ", name), coh*100, round(coh*n_blast), n_blast))
    }

    # Plot
    plots <- list()
    for (name in names(embs)) {
        emb <- embs[[name]]
        if (is.null(emb)) next
        coh <- measure_blast_coherence(emb, labels)

        df <- data.frame(x=emb$V1, y=emb$V2,
                         type=ifelse(labels$label == 1, "AML blast", "Healthy"))
        # Plot healthy first (grey), then blast on top (red)
        df$type <- factor(df$type, levels=c("Healthy", "AML blast"))
        df <- df[order(df$type),]

        p <- ggplot(df, aes(x=x, y=y, color=type, size=type, alpha=type)) +
            geom_point() +
            scale_color_manual(values=c("Healthy"="grey70", "AML blast"="red")) +
            scale_size_manual(values=c("Healthy"=0.2, "AML blast"=1.5)) +
            scale_alpha_manual(values=c("Healthy"=0.3, "AML blast"=1.0)) +
            labs(title=name,
                 subtitle=sprintf("Blast coherence: %.0f%%", coh*100),
                 x="UMAP 1", y="UMAP 2") +
            theme_minimal() +
            theme(legend.position="none",
                  plot.title=element_text(hjust=0.5, face="bold", size=10),
                  plot.subtitle=element_text(hjust=0.5, size=8, color="red"))
        plots[[length(plots)+1]] <- p
    }

    # Shared legend
    df_leg <- data.frame(x=0, y=0, type=factor(c("Healthy","AML blast"), levels=c("Healthy","AML blast")))
    p_leg <- ggplot(df_leg, aes(x=x,y=y,color=type,size=type)) + geom_point() +
        scale_color_manual(values=c("Healthy"="grey70","AML blast"="red"), name="") +
        scale_size_manual(values=c("Healthy"=2,"AML blast"=4), name="") +
        theme_void() + theme(legend.position="bottom")
    legend <- get_legend(p_leg)

    ncols <- length(plots)
    png(sprintf("results/aml_%s_rare.png", pct), width=ncols*450, height=700, res=120)
    grid.arrange(
        do.call(arrangeGrob, c(plots, list(ncol=ncols))),
        legend, nrow=2, heights=c(5,1)
    )
    dev.off()
    cat(sprintf("  Saved: results/aml_%s_rare.png\n", pct))
}
