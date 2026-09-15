script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
out <- dirname(normalizePath(script))
.libPaths(c(file.path(out, "R-library"), .libPaths()))
suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(jsonlite))
write_tsv <- function(x, path) write.table(x, path, sep="\t", quote=FALSE, row.names=FALSE, na="NA")
summary_path <- file.path(out,"influence_summary.json")
summaries <- if(file.exists(summary_path)) read_json(summary_path,simplifyVector=TRUE) else list()
models <- c("primary", "medication_timing", "phase", "usable_bases", "all_579")
requested <- commandArgs(trailingOnly=TRUE)
if (length(requested)>0) {
  stopifnot(all(requested%in%models))
  models <- requested
}
for (name in models) {
  if (!is.null(summaries[[name]])) next
  folder <- file.path(out, name)
  dds <- readRDS(file.path(folder, "dds.rds"))
  x <- model.matrix(design(dds), as.data.frame(colData(dds)))
  hashes <- apply(x, 1, paste, collapse="_")
  eligible <- unname(table(hashes)[hashes])>=3L
  cooks <- assays(dds)[["cooks"]]
  cutoff <- qf(0.99, ncol(x), nrow(x)-ncol(x))
  max_cooks <- apply(cooks, 1, max, na.rm=TRUE)
  leverage <- rowSums(qr.Q(qr(x))^2)
  genes <- data.frame(Geneid=rownames(dds), max_cooks_all_samples=max_cooks,
                       observations_above_review_threshold=rowSums(cooks>cutoff,na.rm=TRUE))
  write_tsv(genes, file.path(folder, "gene_influence.tsv"))
  samples <- data.frame(PATNO=colnames(dds), group=colData(dds)$group,
                         design_leverage=leverage,
                         eligible_for_default_cooks_screen=eligible,
                         median_cooks=apply(cooks,2,median,na.rm=TRUE),
                         fraction_genes_above_review_threshold=colMeans(cooks>cutoff,na.rm=TRUE))
  write_tsv(samples, file.path(folder, "sample_influence.tsv"))
  results <- read.delim(file.path(folder, "results.tsv"))
  ids <- results$Geneid[results$significant_FDR05]
  summaries[[name]] <- list(default_cooks_eligible_samples=sum(eligible),
    descriptive_F99_cutoff=cutoff, genes_above_review_threshold=sum(max_cooks>cutoff),
    significant_genes_above_review_threshold=sum(max_cooks[rownames(dds)%in%ids]>cutoff),
    maximum_design_leverage=max(leverage),
    samples_leverage_above_0_99=sum(leverage>0.99),
    maximum_sample_fraction_genes_above_threshold=max(samples$fraction_genes_above_review_threshold),
    additional_exclusions=0)
  print(c(list(model=name),summaries[[name]]))
  write_json(summaries,summary_path,pretty=TRUE,auto_unbox=TRUE)
  rm(dds,cooks)
  gc()
}
write_json(summaries,file.path(out,"influence_summary.json"),pretty=TRUE,auto_unbox=TRUE)
