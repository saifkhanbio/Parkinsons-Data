script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
out <- dirname(normalizePath(script))
base <- file.path(dirname(out), "ppmi-deseq2-2026-09-12")
.libPaths(c(file.path(base, "R-library"), .libPaths()))
suppressPackageStartupMessages({library(DESeq2); library(jsonlite); library(BiocParallel); library(fgsea)})
options(warn=1)
wt <- function(x,p) write.table(x,p,sep="\t",row.names=FALSE,quote=FALSE,na="NA")
primary <- readRDS(file.path(base,"primary/dds.rds"))
m <- read.delim(file.path(out,"blood_covariates.tsv"), colClasses=c(PATNO="character"),check.names=FALSE)
stopifnot(nrow(m)==445, !anyDuplicated(m$PATNO), nrow(primary)==21888, all(m$PATNO %in% colnames(primary)))
d <- droplevels(as.data.frame(colData(primary))[m$PATNO,,drop=FALSE])
d$sizeFactor <- NULL
d$group <- factor(d$group, levels=c("Control","PD"))
for(k in c("log_wbc_z","neutrophils_z","monocytes_z","eosinophils_z","basophils_z")) d[[k]] <- m[[k]]
stopifnot(identical(as.character(d$group),m$group),sum(d$group=="PD")==303,sum(d$group=="Control")==142)
cts <- counts(primary)[,m$PATNO,drop=FALSE]
stopifnot(all(is.finite(cts)),all(cts>=0),all(cts==round(cts)),!anyDuplicated(rownames(cts)),all(colSums(cts)>0))
forms <- list(subset_reference=~batch+age_c+sex+RIN_c+intergenic_c+group,
 cell_adjusted=~batch+age_c+sex+RIN_c+intergenic_c+log_wbc_z+neutrophils_z+monocytes_z+eosinophils_z+basophils_z+group)
preflight <- list(n=445,PD=303,Control=142,genes=21888,models=list())
for(label in names(forms)) {
 dir.create(file.path(out,label),showWarnings=FALSE)
 x <- model.matrix(forms[[label]],d)
 stopifnot(qr(x)$rank==ncol(x), nrow(x)==445)
 wt(data.frame(PATNO=rownames(x),x,check.names=FALSE),file.path(out,label,"design_matrix.tsv"))
 preflight$models[[label]] <- list(formula=paste(deparse(forms[[label]]),collapse=" "),rank=qr(x)$rank,columns=ncol(x),residual_df=nrow(x)-ncol(x))
}
wt(data.frame(PATNO=rownames(d),d,check.names=FALSE)[,!duplicated(c("PATNO",names(d)))],file.path(out,"colData.tsv"))
wt(data.frame(Geneid=rownames(cts)),file.path(out,"frozen_genes.tsv"))
wt(data.frame(PATNO=colnames(cts),count_sum=colSums(cts)),file.path(out,"count_sums.tsv"))
write_json(preflight,file.path(out,"preflight.json"),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
if("--prepare-only" %in% commandArgs(trailingOnly=TRUE)) quit(status=0)
sets <- readRDS(file.path(base,"msigdb_hallmark_gobp.rds"))
h <- unique(sets[sets$gs_collection=="H" & !is.na(sets$ensembl_gene),c("gs_name","ensembl_gene")])
pathways <- lapply(split(h$ensembl_gene,h$gs_name),unique)
stopifnot(length(pathways)==50)
rm(primary); gc()
for(label in names(forms)) {
 dest <- file.path(out,label)
 cat(format(Sys.time()),"Starting",label,"\n")
 fresh <- DESeqDataSetFromMatrix(cts,d,forms[[label]])
 stopifnot(is.null(sizeFactors(fresh)),is.null(dispersions(fresh)),is.null(normalizationFactors(fresh)))
 fit <- DESeq(fresh,test="Wald",fitType="parametric",sfType="ratio",betaPrior=FALSE,
              minReplicatesForReplace=Inf,parallel=TRUE,BPPARAM=MulticoreParam(8))
 saveRDS(fit,file.path(dest,"initial_fit.rds"),compress=FALSE)
 bad <- which(!mcols(fit)$betaConv)
 if(length(bad)) fit[bad,] <- nbinomWaldTest(fit[bad,],betaPrior=FALSE,maxit=1000)
 ok <- mcols(fit)$betaConv
 stopifnot(!anyNA(ok))
 rr <- results(fit[ok,],name="group_PD_vs_Control",alpha=.05)
 res <- data.frame(Geneid=rownames(fit),beta_converged=ok)
 v <- as.data.frame(rr)[match(res$Geneid,rownames(rr)),,drop=FALSE]
 rownames(v) <- NULL
 res <- cbind(res,v)
 res$padj_no_independent_filter <- p.adjust(res$pvalue,"BH")
 res$dispersion <- dispersions(fit)
 cooks <- assays(fit)[["cooks"]]
 res$max_cooks <- apply(cooks,1,max,na.rm=TRUE)
 x <- model.matrix(forms[[label]],d)
 res$cook_review_flag <- res$max_cooks > qf(.99,ncol(x),nrow(x)-ncol(x))
 wt(res,file.path(dest,"results.tsv"))
 wt(data.frame(PATNO=colnames(fit),size_factor=sizeFactors(fit)),file.path(dest,"size_factors.tsv"))
 saveRDS(fit,file.path(dest,"dds.rds"),compress=FALSE)
 pdf(file.path(dest,"diagnostics.pdf")); plotDispEsts(fit); plotMA(rr,alpha=.05); dev.off()
 write_json(list(status="COMPLETE",n=ncol(fit),initial_nonconverged=length(bad),nonconverged=sum(!ok),
  significant_FDR05=sum(res$padj<.05,na.rm=TRUE),higher_PD=sum(res$padj<.05 & res$log2FoldChange>0,na.rm=TRUE),
  lower_PD=sum(res$padj<.05 & res$log2FoldChange<0,na.rm=TRUE),cook_flagged_significant=sum(res$cook_review_flag & res$padj<.05,na.rm=TRUE),
  fit_type=attr(dispersionFunction(fit),"fitType")),file.path(dest,"model_summary.json"),pretty=TRUE,auto_unbox=TRUE)
 rank <- res[is.finite(res$stat)&is.finite(res$pvalue)&res$beta_converged,c("Geneid","stat")]
 rank$ensembl_id <- sub("\\.[0-9]+$","",rank$Geneid)
 stopifnot(!anyDuplicated(rank$ensembl_id))
 rank <- rank[order(-rank$stat,rank$ensembl_id),]
 wt(rank,file.path(dest,"enrichment_ranks.tsv"))
 set.seed(20260912)
 gsea <- as.data.frame(fgseaMultilevel(pathways=pathways,stats=setNames(rank$stat,rank$ensembl_id),
  minSize=15,maxSize=500,eps=0,sampleSize=101,nPermSimple=10000,scoreType="std",BPPARAM=SerialParam()))
 stopifnot(nrow(gsea)==50,all(is.finite(gsea$NES)),all(is.finite(gsea$padj)))
 gsea$leadingEdge <- vapply(gsea$leadingEdge,paste,collapse=";",character(1))
 wt(gsea[order(gsea$padj),],file.path(dest,"Hallmark_enrichment.tsv"))
 rm(fit,fresh,cooks); gc()
}
cat("Paired models and Hallmark enrichment complete.\n")
