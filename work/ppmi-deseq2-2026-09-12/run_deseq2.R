# Run from any working directory; all paths derive from this script.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
out <- dirname(normalizePath(script))
work <- dirname(out)
.libPaths(c(file.path(out, "R-library"), .libPaths()))
suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(BiocParallel))
set.seed(20260912)
options(warn=1)
workers <- as.integer(Sys.getenv("DESEQ_WORKERS", "8"))
bp <- MulticoreParam(workers=workers, progressbar=FALSE, RNGseed=20260912)
write_tsv <- function(x, path) write.table(x, path, sep="\t", quote=FALSE, row.names=FALSE, na="NA")
status <- function(stage, model=NULL) {
  write_json(list(stage=stage, model=model, time=format(Sys.time(), tz="UTC", usetz=TRUE)),
             file.path(out, "status.json"), pretty=TRUE, auto_unbox=TRUE)
  cat(format(Sys.time()), stage, model, "\n")
}
status("VALIDATING_INPUTS")
qc <- file.path(work, "ppmi-expression-qc-2026-09-12")
counts_all <- as.matrix(read.delim(gzfile(file.path(qc, "raw_counts.tsv.gz")),
                                  row.names=1, check.names=FALSE))
stopifnot(nrow(counts_all)==58780, ncol(counts_all)==579,
          !anyDuplicated(rownames(counts_all)), !anyDuplicated(colnames(counts_all)),
          all(is.finite(counts_all)), all(counts_all>=0),
          all(counts_all==floor(counts_all)), all(colSums(counts_all)>0))
storage.mode(counts_all) <- "integer"
primary <- read.delim(file.path(work, "ppmi-model-design-2026-09-12", "DESeq2_colData.tsv"),
                      colClasses=c(PATNO="character"), check.names=FALSE)
all_data <- read.delim(file.path(out, "all_sample_covariates.tsv"), colClasses=c(PATNO="character"))
manifest <- fromJSON(file.path(qc, "cohort_after_QC.json"))
stopifnot(identical(primary$PATNO, as.character(manifest$PATNO)),
          identical(colnames(counts_all), all_data$PATNO))
keep <- rowSums(counts_all[,primary$PATNO,drop=FALSE]>=10L)>=179L
write_tsv(data.frame(Geneid=rownames(counts_all),
                    primary_samples_count_at_least_10=rowSums(counts_all[,primary$PATNO]>=10L),
                    prefilter_pass=keep), file.path(out, "gene_prefilter.tsv"))
counts_all <- counts_all[keep,,drop=FALSE]
prepare <- function(d) {
  rownames(d) <- d$PATNO
  d$group <- factor(d$group, levels=c("Control", "PD"))
  d$sex <- factor(d$sex, levels=c("Female", "Male"))
  d$phase <- factor(d$phase)
  d$batch <- factor(d$batch)
  for (pair in list(c("age_collection_years", "age_c"), c("RIN", "RIN_c"),
                    c("intergenic_percent", "intergenic_c"), c("usable_percent", "usable_c"))) {
    d[[pair[2]]] <- d[[pair[1]]] - mean(d[[pair[1]]])
  }
  d
}
f <- ~ batch + age_c + sex + RIN_c + intergenic_c + group
models <- list(
  primary=list(data=primary, formula=f),
  medication_timing=list(data=primary[tolower(as.character(primary$medication_timing_sensitivity_exclude))=="false",], formula=f),
  phase=list(data=primary, formula=~phase + age_c + sex + RIN_c + intergenic_c + group),
  usable_bases=list(data=primary, formula=~batch + age_c + sex + RIN_c + usable_c + group),
  all_579=list(data=all_data, formula=f)
)
expected_n <- c(primary=558L, medication_timing=554L, phase=558L, usable_bases=558L, all_579=579L)
design_checks <- list()
for (name in names(models)) {
  d <- prepare(models[[name]]$data)
  x <- model.matrix(models[[name]]$formula, d)
  stopifnot(nrow(d)==expected_n[[name]], nrow(x)==nrow(d), all(is.finite(x)),
            qr(x)$rank==ncol(x), !anyDuplicated(d$PATNO))
  models[[name]]$data <- d
  write_tsv(d, file.path(out, paste0(name, "_colData.tsv")))
  design_checks[[name]] <- list(n=nrow(d), groups=as.list(table(d$group)),
    formula=paste(deparse(models[[name]]$formula), collapse=" "),
    columns=ncol(x), rank=qr(x)$rank, residual_df=nrow(x)-qr(x)$rank)
}
write_json(design_checks, file.path(out, "design_checks.json"), pretty=TRUE, auto_unbox=TRUE)
cat("Genes retained:", nrow(counts_all), "\n")
summaries <- list()
for (name in names(models)) {
  folder <- file.path(out, name)
  dir.create(folder, showWarnings=FALSE)
  if (file.exists(file.path(folder, "summary.json"))) {
    summaries[[name]] <- read_json(file.path(folder, "summary.json"), simplifyVector=TRUE)
    cat("Using completed model:", name, "\n")
    next
  }
  status("FITTING", name)
  d <- models[[name]]$data
  cts <- counts_all[,d$PATNO,drop=FALSE]
  stopifnot(identical(colnames(cts), rownames(d)))
  checkpoint <- file.path(folder, "initial_fit.rds")
  if (file.exists(checkpoint)) {
    dds <- readRDS(checkpoint)
    stopifnot(identical(rownames(dds), rownames(cts)), identical(colnames(dds), colnames(cts)))
  } else {
    dds <- DESeqDataSetFromMatrix(cts, d, models[[name]]$formula)
    dds <- DESeq(dds, test="Wald", fitType="parametric", sfType="ratio",
                 betaPrior=FALSE, minReplicatesForReplace=Inf,
                 parallel=workers>1, BPPARAM=bp)
    saveRDS(dds, checkpoint, compress=FALSE)
  }
  stopifnot("group_PD_vs_Control" %in% resultsNames(dds))
  # Retry numerically difficult fits with more iterations, without changing the model.
  initial_nonconverged <- sum(!mcols(dds)$betaConv)
  if (initial_nonconverged>0) {
    message("Retrying ", initial_nonconverged, " nonconverged genes with maxit=1000.")
    bad <- which(!mcols(dds)$betaConv)
    retried <- nbinomWaldTest(dds[bad,], betaPrior=FALSE, maxit=1000)
    dds[bad,] <- retried
  }
  converged <- mcols(dds)$betaConv
  # Failed numerical fits are explicitly untestable. Recompute BH/independent
  # filtering on converged fits, preserving all original rows in the export.
  res_ok <- results(dds[converged,], name="group_PD_vs_Control", alpha=0.05,
                 independentFiltering=TRUE, pAdjustMethod="BH")
  fixed_ok <- results(dds[converged,], name="group_PD_vs_Control", alpha=0.05,
                   independentFiltering=FALSE, pAdjustMethod="BH")
  res <- results(dds, name="group_PD_vs_Control", alpha=0.05, independentFiltering=FALSE)
  res[,] <- NA_real_
  res[converged,] <- res_ok
  res$baseMean <- mcols(dds)$baseMean
  fixed_padj <- rep(NA_real_, nrow(dds))
  fixed_padj[converged] <- fixed_ok$padj
  tab <- data.frame(Geneid=rownames(res), ensembl_id=sub("\\.[0-9]+$", "", rownames(res)),
                    as.data.frame(res), padj_no_independent_filter=fixed_padj,
                    beta_converged=converged, dispersion=dispersions(dds),
                    max_cooks=mcols(dds)$maxCooks)
  tab$result_status <- ifelse(is.na(tab$pvalue), "cooks_screened_or_unavailable",
                              ifelse(is.na(tab$padj), "independent_filtered", "tested"))
  tab$result_status[!tab$beta_converged] <- "nonconverged_untestable"
  tab$significant_FDR05 <- !is.na(tab$padj) & tab$padj<0.05
  tab <- tab[order(tab$padj, tab$pvalue, na.last=TRUE),]
  write_tsv(tab, file.path(folder, "results.tsv"))
  write_tsv(tab[tab$significant_FDR05,], file.path(folder, "significant_FDR05.tsv"))
  write_tsv(data.frame(PATNO=colnames(dds), size_factor=sizeFactors(dds)),
            file.path(folder, "size_factors.tsv"))
  saveRDS(dds, file.path(folder, "dds.rds"), compress=FALSE)
  summary <- c(design_checks[[name]], list(genes_prefiltered=nrow(dds),
    genes_with_pvalue=sum(!is.na(tab$pvalue)), genes_with_padj=sum(!is.na(tab$padj)),
    significant_FDR05=sum(tab$significant_FDR05),
    higher_in_PD=sum(tab$significant_FDR05 & tab$log2FoldChange>0),
    lower_in_PD=sum(tab$significant_FDR05 & tab$log2FoldChange<0),
    significant_no_independent_filter=sum(fixed_padj<0.05,na.rm=TRUE),
    initial_nonconverged=initial_nonconverged, final_nonconverged=sum(!converged),
    pvalue_unavailable=sum(is.na(tab$pvalue)),
    independent_filtered=sum(!is.na(tab$pvalue)&is.na(tab$padj)),
    independent_filter_threshold=as.numeric(metadata(res_ok)$filterThreshold),
    dispersion_fit=attr(dispersionFunction(dds), "fitType"),
    size_factor_range=range(sizeFactors(dds)), DESeq2_version=as.character(packageVersion("DESeq2"))))
  # Complete marker is written only after all model artifacts exist.
  pdf(file.path(folder, "diagnostics.pdf"), width=10, height=8)
  par(mfrow=c(2,2))
  plotMA(res, alpha=0.05, main=paste(name, "PD vs Control"))
  plotDispEsts(dds, main="Dispersion estimates")
  hist(tab$pvalue, breaks=50, main="Wald p values", xlab="p value", col="grey80")
  plot(tab$log2FoldChange, -log10(pmax(tab$pvalue, .Machine$double.xmin)),
       pch=16, cex=0.3, col=ifelse(tab$significant_FDR05,"#D86738","#999999"),
       xlab="Unshrunk log2 fold change (PD / Control)", ylab="-log10(p)", main="Volcano plot")
  dev.off()
  write_json(summary, file.path(folder, "summary.json"), pretty=TRUE, auto_unbox=TRUE)
  summaries[[name]] <- summary
  print(summary)
  rm(dds, res, res_ok, fixed_ok, tab, cts)
  gc()
}
write_json(summaries, file.path(out, "model_summaries.json"), pretty=TRUE, auto_unbox=TRUE)
capture.output(sessionInfo(), file=file.path(out, "R_session_info.txt"))
status("DESEQ2_COMPLETE")
