script <- sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1])
out <- dirname(normalizePath(script))
analysis <- file.path(dirname(out),"ppmi-deseq2-2026-09-12")
.libPaths(c(file.path(analysis,"R-library"),.libPaths()))
suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(fgsea))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(BiocParallel))
options(warn=1)
write_tsv <- function(x,path)write.table(x,path,sep="\t",quote=FALSE,row.names=FALSE,na="NA")
inputs <- c(file.path(analysis,"primary/dds.rds"),
            file.path(dirname(out),"ppmi-expression-qc-2026-09-12/QC_disposition.tsv"),
            file.path(analysis,"msigdb_hallmark_gobp.rds"),
            file.path(analysis,"c2_c5bp/C2_gene_sets.rds"))
hashes <- tools::md5sum(inputs)
manifest <- file.path(out,"input_md5.tsv")
if(file.exists(manifest)) {
  old <- read.delim(manifest)
  stopifnot(identical(old$path,inputs),identical(old$md5,unname(hashes)))
} else write_tsv(data.frame(path=inputs,md5=unname(hashes)),manifest)
primary <- readRDS(inputs[1])
private_config <- fromJSON(Sys.getenv("PPMI_PRIVATE_CONFIG"))
omitted <- as.character(private_config$qc_sensitivity_exclusions)
stopifnot(length(omitted) > 0L, !anyDuplicated(omitted))
stopifnot(all(omitted %in% colnames(primary)),ncol(primary)==558,nrow(primary)==21888)
keep <- !colnames(primary)%in%omitted
d <- droplevels(as.data.frame(colData(primary))[keep,,drop=FALSE])
f <- design(primary)
d$sizeFactor <- NULL
x <- model.matrix(f,d)
stopifnot(nrow(x)==556,qr(x)$rank==ncol(x),sum(d$group=="PD")==378,
          sum(d$group=="Control")==178,identical(rownames(d),colnames(primary)[keep]))
fresh <- DESeqDataSetFromMatrix(counts(primary)[,keep,drop=FALSE],d,f)
stopifnot(is.null(sizeFactors(fresh)),is.null(normalizationFactors(fresh)),
          is.null(dispersions(fresh)),identical(assayNames(fresh),"counts"))
write_tsv(data.frame(Geneid=rownames(fresh)),file.path(out,"frozen_genes.tsv"))
write_tsv(data.frame(PATNO=colnames(fresh),count_sum=colSums(counts(fresh))),file.path(out,"count_sums.tsv"))
write_tsv(d,file.path(out,"colData.tsv"))
write_tsv(data.frame(PATNO=rownames(x),x,check.names=FALSE),file.path(out,"design_matrix.tsv"))
qc <- read.delim(inputs[2],colClasses=c(PATNO="character"))
write_tsv(qc[qc$PATNO%in%omitted,],file.path(out,"omitted_sample_evidence.tsv"))
write_json(list(status="VALIDATED",n=ncol(fresh),groups=as.list(table(d$group)),
                genes=nrow(fresh),omitted_samples=omitted,design_columns=ncol(x),rank=qr(x)$rank,
                formula=paste(deparse(f),collapse=" "),inherited_size_factors=FALSE,inherited_dispersions=FALSE),
           file.path(out,"preflight.json"),pretty=TRUE,auto_unbox=TRUE)
rm(primary);gc()
if("--prepare-only"%in%commandArgs(trailingOnly=TRUE))quit(status=0)
checkpoint <- file.path(out,"initial_fit.rds")
if(file.exists(checkpoint)) {
  fitted <- readRDS(checkpoint)
  stopifnot(identical(colnames(fitted),colnames(fresh)),identical(rownames(fitted),rownames(fresh)),
            identical(counts(fitted),counts(fresh)),identical(design(fitted),f))
  rm(fresh)
} else {
  fitted <- DESeq(fresh,test="Wald",fitType="parametric",sfType="ratio",betaPrior=FALSE,
                  minReplicatesForReplace=Inf,parallel=TRUE,BPPARAM=MulticoreParam(8))
  rm(fresh)
  saveRDS(fitted,checkpoint,compress=FALSE)
}
bad <- which(!mcols(fitted)$betaConv)
if(length(bad)>0)fitted[bad,] <- nbinomWaldTest(fitted[bad,],betaPrior=FALSE,maxit=1000)
ok <- mcols(fitted)$betaConv
stopifnot(!anyNA(ok))
r <- results(fitted[ok,],name="group_PD_vs_Control",alpha=.05)
res <- merge(data.frame(Geneid=rownames(fitted),beta_converged=ok),
             data.frame(Geneid=rownames(r),as.data.frame(r)),by="Geneid",all.x=TRUE,sort=FALSE)
res$dispersion <- dispersions(fitted)[match(res$Geneid,rownames(fitted))]
write_tsv(res,file.path(out,"results.tsv"))
write_tsv(data.frame(PATNO=colnames(fitted),size_factor=sizeFactors(fitted)),file.path(out,"size_factors.tsv"))
saveRDS(fitted,file.path(out,"dds.rds"),compress=FALSE)
write_json(list(status="COMPLETE",n=ncol(fitted),genes=nrow(fitted),nonconverged=sum(!ok),
                initial_nonconverged=length(bad),significant_FDR05=sum(res$padj<.05,na.rm=TRUE),
                higher_in_PD=sum(res$padj<.05 & res$log2FoldChange>0,na.rm=TRUE),
                lower_in_PD=sum(res$padj<.05 & res$log2FoldChange<0,na.rm=TRUE)),
           file.path(out,"model_summary.json"),pretty=TRUE,auto_unbox=TRUE)
ranks <- res[res$beta_converged & is.finite(res$stat) & is.finite(res$pvalue),c("Geneid","stat")]
ranks$ensembl_id <- sub("\\.[0-9]+$","",ranks$Geneid)
ranks <- ranks[order(-ranks$stat,ranks$ensembl_id),]
stopifnot(!anyDuplicated(ranks$ensembl_id))
write_tsv(ranks,file.path(out,"enrichment_ranks.tsv"))
stats <- setNames(ranks$stat,ranks$ensembl_id)
old <- readRDS(inputs[3]); c2 <- readRDS(inputs[4])
stopifnot(identical(unique(old$db_version),unique(c2$db_version)))
sets <- list(Hallmark=old[old$gs_collection=="H",],C2=c2,
             C5_BP=old[old$gs_collection=="C5" & old$gs_subcollection=="GO:BP",])
metadata <- list()
for(label in names(sets)) {
  z <- sets[[label]]
  pairs <- unique(z[!is.na(z$ensembl_gene)&nzchar(z$ensembl_gene),c("gs_name","ensembl_gene")])
  pathways <- lapply(split(pairs$ensembl_gene,pairs$gs_name),unique)
  sizes <- vapply(pathways,function(v)length(intersect(v,names(stats))),integer(1))
  included <- sizes>=15 & sizes<=500
  write_tsv(data.frame(pathway=names(sizes),measured_genes=unname(sizes),tested=included),
            file.path(out,paste0(label,"_coverage.tsv")))
  destination <- file.path(out,paste0(label,"_enrichment.tsv"))
  if(!file.exists(destination)) {
    cat("Fitting",label,"with",sum(included),"sets\n")
    set.seed(20260912)
    enrichment <- fgseaMultilevel(pathways=pathways,stats=stats,minSize=15,maxSize=500,
        eps=0,sampleSize=101,nPermSimple=10000,scoreType="std",BPPARAM=SerialParam())
    enrichment <- as.data.frame(enrichment[order(enrichment$padj,enrichment$pval),])
    enrichment$leadingEdge <- vapply(enrichment$leadingEdge,paste,collapse=";",character(1))
    write_tsv(enrichment,paste0(destination,".tmp"))
    stopifnot(file.rename(paste0(destination,".tmp"),destination))
  } else enrichment <- read.delim(destination)
  stopifnot(!anyDuplicated(enrichment$pathway),identical(sort(enrichment$pathway),sort(names(sizes)[included])),
            all(enrichment$size==sizes[enrichment$pathway]),all(is.finite(enrichment$NES)),
            all(is.finite(enrichment$pval)),all(is.finite(enrichment$padj)))
  metadata[[label]] <- unique(z[,c("gs_name","gs_collection","gs_subcollection","gs_description","gs_url")])
  metadata[[label]]$collection <- label
}
write_tsv(do.call(rbind,metadata),file.path(out,"pathway_metadata.tsv"))
stopifnot(identical(hashes,tools::md5sum(inputs)))
write_json(list(status="COMPLETE",ranked_genes=length(stats),db_version=unique(c2$db_version),
                seed=20260912,minSize=15,maxSize=500,sampleSize=101,nPermSimple=10000,eps=0,
                fgsea_version=as.character(packageVersion("fgsea"))),
           file.path(out,"enrichment_summary.json"),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
cat("QC sensitivity fitting and enrichment complete.\n")
