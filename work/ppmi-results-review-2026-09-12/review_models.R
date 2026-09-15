script <- sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1])
out <- dirname(normalizePath(script)); analysis <- file.path(dirname(out),"ppmi-deseq2-2026-09-12")
.libPaths(c(file.path(analysis,"R-library"),.libPaths()))
suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(BiocParallel))
options(warn=1)
write_tsv <- function(x,path)write.table(x,path,sep="\t",quote=FALSE,row.names=FALSE,na="NA")
dds <- readRDS(file.path(analysis,"primary/dds.rds"))
d <- as.data.frame(colData(dds)); x <- model.matrix(design(dds),d)
write_tsv(data.frame(PATNO=rownames(x),x,check.names=FALSE),file.path(out,"primary_design_matrix.tsv"))
leverage <- rowSums(qr.Q(qr(x))^2)
cooks <- assays(dds)[["cooks"]]
cutoff <- qf(.99,ncol(x),nrow(x)-ncol(x))
flags <- which(cooks>cutoff,arr.ind=TRUE)
results <- read.delim(file.path(analysis,"primary/results_annotated.tsv"))
sig <- results$Geneid[tolower(as.character(results$significant_FDR05))=="true"]
pairs <- data.frame(Geneid=rownames(dds)[flags[,1]],PATNO=colnames(dds)[flags[,2]],
                     cooks=cooks[flags],design_leverage=leverage[flags[,2]],
                     group=d$group[flags[,2]],batch=d$batch[flags[,2]],
                     raw_count=counts(dds)[flags],normalized_count=counts(dds,normalized=TRUE)[flags])
pairs$primary_significant <- pairs$Geneid%in%sig
write_tsv(pairs,file.path(out,"flagged_gene_sample_pairs.tsv"))
sample_review <- merge(read.delim(file.path(analysis,"primary/sample_influence.tsv")),
                       data.frame(PATNO=d$PATNO,batch=d$batch,RIN=d$RIN,intergenic_percent=d$intergenic_percent),by="PATNO",sort=FALSE)
sample_review$flagged_genes <- as.integer(table(factor(pairs$PATNO,levels=sample_review$PATNO)))
sample_review$flagged_significant_genes <- as.integer(table(factor(pairs$PATNO[pairs$primary_significant],levels=sample_review$PATNO)))
write_tsv(sample_review[order(-sample_review$flagged_genes),],file.path(out,"sample_influence_review.tsv"))
target <- pairs[pairs$primary_significant,]
stopifnot(length(unique(target$Geneid))==11,nrow(target)>0)
loo <- list()
for(i in seq_len(nrow(target))) {
  gene <- target$Geneid[i]; omitted <- target$PATNO[i]
  keep <- colnames(dds)!=omitted
  cd <- droplevels(d[keep,])
  one <- DESeqDataSetFromMatrix(counts(dds)[gene,keep,drop=FALSE],cd,design(dds))
  sizeFactors(one) <- sizeFactors(dds)[keep]
  dispersions(one) <- dispersions(dds)[match(gene,rownames(dds))]
  one <- nbinomWaldTest(one,betaPrior=FALSE,maxit=1000,quiet=TRUE)
  r <- results(one,name="group_PD_vs_Control",independentFiltering=FALSE,cooksCutoff=FALSE)
  original <- results[match(gene,results$Geneid),]
  loo[[i]] <- data.frame(Geneid=gene,PATNO_omitted=omitted,cooks=target$cooks[i],
                          design_leverage=target$design_leverage[i],original_log2FC=original$log2FoldChange,
                          original_SE=original$lfcSE,loo_log2FC=r$log2FoldChange,loo_SE=r$lfcSE,
                          converged=mcols(one)$betaConv,delta_log2FC=r$log2FoldChange-original$log2FoldChange)
}
write_tsv(do.call(rbind,loo),file.path(out,"flagged_gene_leave_one_out.tsv"))
rm(cooks);gc()

# The additional nuisance design uses explicit matched site and self-report categories.
extra <- read.delim(file.path(out,"confounder_colData.tsv"),colClasses=c(PATNO="character",site="character"))
stopifnot(identical(d$PATNO,extra$PATNO),!anyNA(extra$site),!anyNA(extra$race_category))
d$site <- factor(extra$site)
d$race_category <- relevel(factor(extra$race_category),ref="white_only")
f <- ~batch+age_c+sex+RIN_c+intergenic_c+site+race_category+group
xx <- model.matrix(f,d)
full_x <- xx
nuisance <- xx[,colnames(xx)!="groupPD",drop=FALSE]
q <- qr(nuisance)
retained <- sort(q$pivot[seq_len(q$rank)])
xx <- cbind(nuisance[,retained,drop=FALSE],groupPD=xx[,"groupPD"])
# Remove only redundant nuisance columns; preserve the diagnosis coefficient.
stopifnot(max(abs(full_x-xx%*%qr.coef(qr(xx),full_x)))<1e-8)
check <- list(n=nrow(xx),columns=ncol(xx),rank=qr(xx)$rank,full_rank=qr(xx)$rank==ncol(xx),
              original_columns=ncol(full_x),redundant_nuisance_columns=setdiff(colnames(full_x),colnames(xx)),
              formula=paste(deparse(f),collapse=" "),site_levels=nlevels(d$site),race_counts=as.list(table(d$race_category)))
write_json(check,file.path(out,"site_race_design_check.json"),pretty=TRUE,auto_unbox=TRUE)
stopifnot(check$full_rank)
write_tsv(data.frame(PATNO=rownames(xx),xx,check.names=FALSE),file.path(out,"site_race_design_matrix.tsv"))
write_tsv(d,file.path(out,"site_race_colData.tsv"))
checkpoint <- file.path(out,"site_race_initial_fit.rds")
if(file.exists(checkpoint)) {
  adjusted <- readRDS(checkpoint)
} else {
  adjusted <- DESeqDataSetFromMatrix(counts(dds),d,xx)
  rm(dds);gc()
  adjusted <- DESeq(adjusted,test="Wald",fitType="parametric",sfType="ratio",betaPrior=FALSE,
                    minReplicatesForReplace=Inf,parallel=TRUE,BPPARAM=MulticoreParam(8))
  saveRDS(adjusted,checkpoint,compress=FALSE)
}
bad <- which(!mcols(adjusted)$betaConv)
if(length(bad)>0) {
  fitted_matrix <- attr(adjusted,"modelMatrix")
  stopifnot(is.matrix(fitted_matrix),identical(dim(fitted_matrix),dim(xx)),
            max(abs(fitted_matrix-xx))<1e-10)
  adjusted[bad,] <- nbinomWaldTest(adjusted[bad,],betaPrior=FALSE,maxit=1000,
                                  modelMatrix=fitted_matrix)
}
ok <- mcols(adjusted)$betaConv
r <- results(adjusted[ok,],name="groupPD",alpha=.05)
tab <- data.frame(Geneid=rownames(adjusted),beta_converged=ok)
tab <- merge(tab,data.frame(Geneid=rownames(r),as.data.frame(r)),by="Geneid",all.x=TRUE,sort=FALSE)
write_tsv(tab,file.path(out,"site_race_results.tsv"))
saveRDS(adjusted,file.path(out,"site_race_dds.rds"),compress=FALSE)
write_json(list(status="COMPLETE",n=ncol(adjusted),genes=nrow(adjusted),nonconverged=sum(!ok),
                 significant_FDR05=sum(tab$padj<.05,na.rm=TRUE),higher_in_PD=sum(tab$padj<.05 & tab$log2FoldChange>0,na.rm=TRUE),
                 lower_in_PD=sum(tab$padj<.05 & tab$log2FoldChange<0,na.rm=TRUE)),
           file.path(out,"model_summary.json"),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
