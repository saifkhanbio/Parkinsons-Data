script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
out <- dirname(normalizePath(script))
.libPaths(c(file.path(out, "R-library"), .libPaths()))
suppressPackageStartupMessages(library(fgsea))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(BiocParallel))
options(warn=1)
gene_sets <- readRDS(file.path(out, "msigdb_hallmark_gobp.rds"))
write_tsv <- function(x, path) write.table(x, path, sep="\t", quote=FALSE, row.names=FALSE, na="NA")
models <- c("primary", "medication_timing", "phase", "usable_bases", "all_579")
requested <- commandArgs(trailingOnly=TRUE)
if (length(requested)>0) {
  stopifnot(all(requested%in%models))
  models <- requested
}
summary_path <- file.path(out, "enrichment_summary.json")
summary <- if(file.exists(summary_path)) read_json(summary_path, simplifyVector=TRUE) else list()
for (model in models) {
  stopifnot(file.exists(file.path(out, model, "summary.json")))
  results <- read.delim(file.path(out, model, "results.tsv"))
  ranks <- results[is.finite(results$stat) & is.finite(results$pvalue) & results$beta_converged,]
  # Prefer the largest mean count if version stripping ever creates duplicates.
  # This rule is independent of the association statistic.
  ranks <- ranks[order(-ranks$baseMean, ranks$Geneid),]
  duplicates <- sum(duplicated(ranks$ensembl_id))
  ranks <- ranks[!duplicated(ranks$ensembl_id),]
  ranks <- ranks[order(-ranks$stat, ranks$ensembl_id),]
  stats <- setNames(ranks$stat, ranks$ensembl_id)
  stopifnot(!anyDuplicated(names(stats)), all(is.finite(stats)))
  write_tsv(ranks[,c("Geneid", "ensembl_id", "stat", "baseMean")],
            file.path(out, model, "enrichment_ranks.tsv"))
  collections <- if(model=="primary") c("H", "GO:BP") else "H"
  for (collection in collections) {
    label <- if(collection=="H") "hallmark" else "gobp"
    key <- paste(model,label,sep="_")
    if (!is.null(summary[[key]]) && file.exists(file.path(out,model,paste0(label,"_enrichment.tsv")))) next
    selected <- if(collection=="H") gene_sets$gs_collection=="H" else gene_sets$gs_subcollection=="GO:BP"
    selected[is.na(selected)] <- FALSE
    pairs <- gene_sets[selected,c("gs_name", "ensembl_gene")]
    pairs <- unique(pairs[!is.na(pairs$ensembl_gene)&nzchar(pairs$ensembl_gene),])
    pathways <- lapply(split(pairs$ensembl_gene, pairs$gs_name), unique)
    set.seed(20260912)
    cat(format(Sys.time()), model, label, "ranked genes:", length(stats), "\n")
    res <- fgseaMultilevel(pathways=pathways, stats=stats, minSize=15, maxSize=500,
                           eps=0, sampleSize=101, nPermSimple=10000,
                           scoreType="std", BPPARAM=SerialParam())
    res <- as.data.frame(res[order(res$padj, res$pval),])
    res$leadingEdge <- vapply(res$leadingEdge, paste, collapse=";", character(1))
    write_tsv(res, file.path(out, model, paste0(label, "_enrichment.tsv")))
    summary[[paste(model,label,sep="_")]] <- list(model=model, collection=label,
      ranked_genes=length(stats), duplicate_stable_ids_removed=duplicates,
      ranked_genes_in_collection=sum(names(stats)%in%pairs$ensembl_gene),
      gene_sets_tested=nrow(res), pvalues_unavailable=sum(is.na(res$pval)),
      significant_FDR05=sum(res$padj<0.05, na.rm=TRUE),
      positive_NES_FDR05=sum(res$padj<0.05 & res$NES>0,na.rm=TRUE),
      negative_NES_FDR05=sum(res$padj<0.05 & res$NES<0,na.rm=TRUE))
    print(summary[[paste(model,label,sep="_")]])
    write_json(summary, summary_path, pretty=TRUE, auto_unbox=TRUE)
  }
}
write_json(summary, file.path(out, "enrichment_summary.json"), pretty=TRUE, auto_unbox=TRUE)
capture.output(sessionInfo(), file=file.path(out, "enrichment_R_session_info.txt"))
