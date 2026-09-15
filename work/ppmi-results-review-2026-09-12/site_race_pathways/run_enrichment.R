script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
out <- dirname(normalizePath(script))
review <- dirname(out)
analysis <- file.path(dirname(review), "ppmi-deseq2-2026-09-12")
.libPaths(c(file.path(analysis,"R-library"),.libPaths()))
suppressPackageStartupMessages(library(fgsea))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(BiocParallel))
options(warn=1)
write_tsv <- function(x,path) write.table(x,path,sep="\t",quote=FALSE,row.names=FALSE,na="NA")
inputs <- c(file.path(review,"site_race_results.tsv"),
            file.path(analysis,"msigdb_hallmark_gobp.rds"),
            file.path(analysis,"c2_c5bp/C2_gene_sets.rds"))
hashes <- tools::md5sum(inputs)
manifest <- file.path(out,"input_md5.tsv")
if(file.exists(manifest)) {
  old <- read.delim(manifest)
  stopifnot(identical(old$path,inputs),identical(old$md5,unname(hashes)))
} else write_tsv(data.frame(path=inputs,md5=unname(hashes)),manifest)
tab <- read.delim(inputs[1])
eligible <- tab$beta_converged & is.finite(tab$stat) & is.finite(tab$pvalue)
ranks <- tab[eligible,c("Geneid","stat")]
ranks$ensembl_id <- sub("\\.[0-9]+$","",ranks$Geneid)
ranks <- ranks[order(-ranks$stat,ranks$ensembl_id),]
stopifnot(nrow(ranks)==21880,!anyDuplicated(ranks$ensembl_id))
write_tsv(ranks,file.path(out,"enrichment_ranks.tsv"))
stats <- setNames(ranks$stat,ranks$ensembl_id)
old <- readRDS(inputs[2]); c2 <- readRDS(inputs[3])
stopifnot(identical(unique(old$db_version),unique(c2$db_version)))
sets <- list(Hallmark=old[old$gs_collection=="H",],C2=c2,
             C5_BP=old[old$gs_collection=="C5" & old$gs_subcollection=="GO:BP",])
metadata <- list()
for(label in names(sets)) {
  d <- sets[[label]]
  pairs <- unique(d[!is.na(d$ensembl_gene)&nzchar(d$ensembl_gene),c("gs_name","ensembl_gene")])
  pathways <- lapply(split(pairs$ensembl_gene,pairs$gs_name),unique)
  sizes <- vapply(pathways,function(x)length(intersect(x,names(stats))),integer(1))
  included <- sizes>=15 & sizes<=500
  write_tsv(data.frame(pathway=names(sizes),measured_genes=unname(sizes),tested=included),
            file.path(out,paste0(label,"_coverage.tsv")))
  destination <- file.path(out,paste0(label,"_enrichment.tsv"))
  if(!file.exists(destination)) {
    cat("Fitting",label,"with",sum(included),"sets\n")
    set.seed(20260912)
    res <- fgseaMultilevel(pathways=pathways,stats=stats,minSize=15,maxSize=500,
                           eps=0,sampleSize=101,nPermSimple=10000,scoreType="std",BPPARAM=SerialParam())
    res <- as.data.frame(res[order(res$padj,res$pval),])
    res$leadingEdge <- vapply(res$leadingEdge,paste,collapse=";",character(1))
    write_tsv(res,paste0(destination,".tmp"))
    stopifnot(file.rename(paste0(destination,".tmp"),destination))
  } else res <- read.delim(destination)
  stopifnot(!anyDuplicated(res$pathway),identical(sort(res$pathway),sort(names(sizes)[included])),
            all(res$size==sizes[res$pathway]),all(is.finite(res$NES)),all(is.finite(res$pval)),
            all(is.finite(res$padj)))
  metadata[[label]] <- unique(d[,c("gs_name","gs_collection","gs_subcollection","gs_description","gs_url")])
  metadata[[label]]$collection <- label
}
write_tsv(do.call(rbind,metadata),file.path(out,"pathway_metadata.tsv"))
stopifnot(identical(hashes,tools::md5sum(inputs)))
write_json(list(status="COMPLETE",model="site_race",ranked_genes=length(stats),
  nonconverged_excluded=sum(!tab$beta_converged),db_version=unique(c2$db_version),
  minSize=15,maxSize=500,seed=20260912,nPermSimple=10000,sampleSize=101,eps=0,
  fgsea_version=as.character(packageVersion("fgsea")),
  inference="CPU fgseaMultilevel, SerialParam; CUDA validation follows"),
  file.path(out,"enrichment_summary.json"),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
cat("Enrichment complete.\n")
