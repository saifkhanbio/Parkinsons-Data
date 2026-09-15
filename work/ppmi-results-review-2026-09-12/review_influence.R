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

cat("Influence review complete.\n")
