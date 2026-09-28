# Resume enrichment from saved converged-gene ranks; never refit a model here.
# A Hallmark may become ineligible under the original minSize=15 rule.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
root <- dirname(normalizePath(script))
base <- file.path(dirname(root), "ppmi-deseq2-2026-09-12")
.libPaths(c(file.path(base, "R-library"), .libPaths()))
suppressPackageStartupMessages({library(fgsea); library(BiocParallel); library(jsonlite)})
wt <- function(x,p) write.table(x,p,sep="\t",row.names=FALSE,quote=FALSE,na="NA")
sets <- readRDS(file.path(base,"msigdb_hallmark_gobp.rds"))
h <- unique(sets[sets$gs_collection=="H" & !is.na(sets$ensembl_gene),c("gs_name","ensembl_gene")])
pathways <- lapply(split(h$ensembl_gene,h$gs_name),unique)
stopifnot(length(pathways)==50)
for(stratum in commandArgs(trailingOnly=TRUE)) {
 stopifnot(stratum %in% c("same_month","preceding_months"))
 for(model in c("subset_reference","cell_adjusted")) {
  dest <- file.path(root,stratum,model)
  stopifnot(file.exists(file.path(dest,"model_summary.json")))
  rank <- read.delim(file.path(dest,"enrichment_ranks.tsv"))
  stopifnot(!anyDuplicated(rank$ensembl_id),all(is.finite(rank$stat)))
  coverage <- data.frame(pathway=names(pathways),ranked_members=vapply(pathways,function(x)length(intersect(x,rank$ensembl_id)),integer(1)))
  coverage$eligible <- coverage$ranked_members>=15 & coverage$ranked_members<=500
  wt(coverage,file.path(dest,"Hallmark_coverage.tsv"))
  target <- file.path(dest,"Hallmark_enrichment.tsv")
  if(!file.exists(target)) {
   set.seed(20260912)
   gsea <- as.data.frame(fgseaMultilevel(pathways=pathways,stats=setNames(rank$stat,rank$ensembl_id),
    minSize=15,maxSize=500,eps=0,sampleSize=101,nPermSimple=10000,scoreType="std",BPPARAM=SerialParam()))
   gsea$leadingEdge <- vapply(gsea$leadingEdge,paste,collapse=";",character(1))
   stopifnot(setequal(gsea$pathway,coverage$pathway[coverage$eligible]),all(is.finite(gsea$NES)),all(is.finite(gsea$padj)))
   wt(gsea[order(gsea$padj),],target)
  }
  gsea <- read.delim(target)
  stopifnot(setequal(gsea$pathway,coverage$pathway[coverage$eligible]))
  write_json(list(status="COMPLETE",universe=50,tested=nrow(gsea),excluded_below_minSize=as.character(coverage$pathway[coverage$ranked_members<15]),excluded_above_maxSize=as.character(coverage$pathway[coverage$ranked_members>500])),
   file.path(dest,"enrichment_summary.json"),pretty=TRUE,auto_unbox=TRUE)
  cat(stratum,model,"tested",nrow(gsea),"of 50 Hallmarks\n")
 }
}
