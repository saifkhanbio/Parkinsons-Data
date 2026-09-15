script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
out <- dirname(normalizePath(script))
analysis <- dirname(out)
.libPaths(c(file.path(analysis,"R-library"),.libPaths()))
Sys.setenv(R_USER_CACHE_DIR=file.path(analysis,"cache"))
suppressPackageStartupMessages(library(msigdbr))
suppressPackageStartupMessages(library(fgsea))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(BiocParallel))
options(warn=1)
write_tsv <- function(x,path) write.table(x,path,sep="\t",quote=FALSE,row.names=FALSE,na="NA")
ranks <- read.delim(file.path(analysis,"primary/enrichment_ranks.tsv"))
results <- read.delim(file.path(analysis,"primary/results.tsv"))
eligible <- results[results$beta_converged & is.finite(results$pvalue) & is.finite(results$stat),]
stopifnot(nrow(ranks)==21885, !anyDuplicated(ranks$ensembl_id),
          identical(sort(ranks$Geneid),sort(eligible$Geneid)),
          isTRUE(all.equal(ranks$stat,eligible$stat[match(ranks$Geneid,eligible$Geneid)])))
stats <- setNames(ranks$stat,ranks$ensembl_id)
c2 <- msigdbr(db_species="HS",species="Homo sapiens",collection="C2")
old_sets <- readRDS(file.path(analysis,"msigdb_hallmark_gobp.rds"))
bp <- old_sets[old_sets$gs_collection=="C5" & old_sets$gs_subcollection=="GO:BP",]
stopifnot(identical(unique(c2$db_version),unique(bp$db_version)))
sets <- list(C2=c2,C5_BP=bp)
metadata <- list()
for(label in names(sets)) {
  d <- sets[[label]]
  pairs <- unique(d[!is.na(d$ensembl_gene)&nzchar(d$ensembl_gene),c("gs_name","ensembl_gene")])
  pathways <- lapply(split(pairs$ensembl_gene,pairs$gs_name),unique)
  sizes <- vapply(pathways,function(x)length(intersect(x,names(stats))),integer(1))
  included <- sizes>=15 & sizes<=500
  write_tsv(data.frame(pathway=names(sizes),measured_genes=unname(sizes),tested=included),
            file.path(out,paste0(label,"_coverage.tsv")))
  if(label=="C2") {
    set.seed(20260912)
    res <- fgseaMultilevel(pathways=pathways,stats=stats,minSize=15,maxSize=500,
                           eps=0,sampleSize=101,nPermSimple=10000,scoreType="std",BPPARAM=SerialParam())
    res <- as.data.frame(res[order(res$padj,res$pval),])
    res$leadingEdge <- vapply(res$leadingEdge,paste,collapse=";",character(1))
    write_tsv(res,file.path(out,"C2_enrichment.tsv"))
  } else {
    res <- read.delim(file.path(analysis,"primary/gobp_enrichment.tsv"))
    stopifnot(identical(sort(res$pathway),sort(names(sizes)[included])),
              all(res$size==sizes[res$pathway]))
    stopifnot(file.copy(file.path(analysis,"primary/gobp_enrichment.tsv"),
                        file.path(out,"C5_BP_enrichment.tsv"),overwrite=TRUE))
  }
  stopifnot(!anyDuplicated(res$pathway),nrow(res)==sum(included),all(is.finite(res$NES)))
  metadata[[label]] <- unique(d[,c("gs_name","gs_collection","gs_subcollection","gs_description","gs_url")])
  metadata[[label]]$collection_label <- label
}
write_tsv(do.call(rbind,metadata),file.path(out,"pathway_metadata.tsv"))
saveRDS(c2,file.path(out,"C2_gene_sets.rds"))
write_json(list(model="primary",db_version=unique(c2$db_version),ranked_genes=length(stats),
  C2_gene_sets=length(unique(c2$gs_name)),C5_BP_gene_sets=length(unique(bp$gs_name)),
  C2_subcollections=unique(c2$gs_subcollection),C5_BP_reused=TRUE,
  rank_source="../primary/enrichment_ranks.tsv",minSize=15,maxSize=500,seed=20260912,
  nPermSimple=10000,sampleSize=101,eps=0,scoreType="std",
  fgsea_version=as.character(packageVersion("fgsea")),msigdbr_version=as.character(packageVersion("msigdbr")),
  source="https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp"),
  file.path(out,"provenance.json"),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R_session_info.txt"))
cat("C2 fitted; existing C5:GO:BP results verified and exported.\n")
