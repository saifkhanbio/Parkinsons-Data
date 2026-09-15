script <- sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1])
out <- dirname(normalizePath(script)); work <- dirname(out)
.libPaths(c(file.path(work,"ppmi-deseq2-2026-09-12/R-library"),.libPaths()))
suppressPackageStartupMessages({library(DESeq2);library(jsonlite)})
wt <- function(x,path)write.table(x,path,sep="\t",quote=FALSE,row.names=FALSE,na="NA")
targets <- read.delim(file.path(out,"annotations.tsv"),check.names=FALSE)$Geneid
stopifnot(length(targets)==5,!anyDuplicated(targets))
sources <- c(`528`="ppmi-blood-cell-adjustment-2026-09-12",`445`="ppmi-blood-cell-timing-2026-09-12")
observations <- list(); omissions <- list(); summaries <- list()
for(window in names(sources)) {
 path <- file.path(work,sources[[window]],"cell_adjusted")
 dds <- readRDS(file.path(path,"dds.rds"))
 tab <- read.delim(file.path(path,"results.tsv"))
 d <- as.data.frame(colData(dds)); x <- model.matrix(design(dds),d)
 stopifnot(ncol(dds)==as.integer(window),all(targets %in% rownames(dds)),qr(x)$rank==ncol(x))
 leverage <- rowSums(qr.Q(qr(x))^2)
 threshold <- qf(.99,ncol(x),nrow(x)-ncol(x))
 raw <- counts(dds)[targets,,drop=FALSE]
 norm <- counts(dds,normalized=TRUE)[targets,,drop=FALSE]
 cooks <- assays(dds)[["cooks"]][targets,,drop=FALSE]
 stopifnot(all(is.finite(raw)),all(raw>=0),all(raw==round(raw)),all(is.finite(norm)))
 for(gene in targets) {
  original <- tab[match(gene,tab$Geneid),]
  stopifnot(original$beta_converged)
  cks <- cooks[gene,]
  oi <- data.frame(window=window,Geneid=gene,PATNO=colnames(dds),group=d$group,batch=d$batch,
                   raw_count=raw[gene,],normalized_count=norm[gene,],cooks=cks,
                   design_leverage=leverage,cook_review_threshold=threshold)
  oi$cook_review_flag <- is.finite(oi$cooks)&oi$cooks>threshold
  observations[[length(observations)+1]] <- oi
  finite <- which(is.finite(cks))
  stopifnot(length(finite)>0)
  imax <- finite[which.max(cks[finite])]
  nmax <- which.max(norm[gene,])
  summaries[[length(summaries)+1]] <- data.frame(window=window,Geneid=gene,
    maximum_cooks=cks[imax],threshold=threshold,flagged_observations=sum(oi$cook_review_flag),
    max_cooks_PATNO=colnames(dds)[imax],max_normalized_PATNO=colnames(dds)[nmax])
  for(index in unique(c(imax,nmax))) {
   keep <- seq_len(ncol(dds))!=index
   cd <- droplevels(d[keep,,drop=FALSE]); cd$sizeFactor <- NULL
   xx <- model.matrix(design(dds),cd)
   stopifnot(qr(xx)$rank==ncol(xx))
   one <- DESeqDataSetFromMatrix(raw[gene,keep,drop=FALSE],cd,design(dds))
   sizeFactors(one) <- sizeFactors(dds)[keep]
   dispersions(one) <- dispersions(dds)[match(gene,rownames(dds))]
   one <- nbinomWaldTest(one,betaPrior=FALSE,maxit=1000,quiet=TRUE)
   rr <- results(one,name="group_PD_vs_Control",independentFiltering=FALSE,cooksCutoff=FALSE)
   ok <- isTRUE(mcols(one)$betaConv)
   effect <- if(ok)rr$log2FoldChange else NA_real_
   omissions[[length(omissions)+1]] <- data.frame(window=window,Geneid=gene,PATNO_omitted=colnames(dds)[index],
     reason=paste(c(if(index==imax)"maximum_Cook",if(index==nmax)"maximum_normalized_count"),collapse=";"),
     cooks=cks[index],design_leverage=leverage[index],original_log2FC=original$log2FoldChange,
     original_SE=original$lfcSE,conditional_log2FC=effect,conditional_SE=if(ok)rr$lfcSE else NA_real_,
     converged=ok,delta_log2FC=effect-original$log2FoldChange,
     delta_original_SE_units=(effect-original$log2FoldChange)/original$lfcSE,
     same_direction=if(ok)sign(effect)==sign(original$log2FoldChange) else NA)
  }
 }
 rm(dds);gc()
}
wt(do.call(rbind,observations),file.path(out,"sample_counts_and_influence.tsv"))
wt(do.call(rbind,summaries),file.path(out,"gene_influence_summary.tsv"))
wt(do.call(rbind,omissions),file.path(out,"conditional_omission_checks.tsv"))
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
cat("Targeted influence checks complete.\n")
