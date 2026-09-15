script <- sub('^--file=', '', grep('^--file=',commandArgs(),value=TRUE)[1])
out <- dirname(normalizePath(script))
base <- file.path(dirname(out),'ppmi-deseq2-2026-09-12')
cached <- as.data.frame(readRDS(file.path(base,'msigdb_hallmark_gobp.rds')))
sets <- list(Hallmark=cached[cached$gs_collection=='H',],
             C2=as.data.frame(readRDS(file.path(base,'c2_c5bp/C2_gene_sets.rds'))),
             C5_BP=cached[cached$gs_collection=='C5' & cached$gs_subcollection=='GO:BP',])
files <- c(Hallmark='primary/hallmark_enrichment.tsv',C2='c2_c5bp/C2_enrichment.tsv',
           C5_BP='c2_c5bp/C5_BP_enrichment.tsv')
expected <- c(Hallmark=25,C2=1505,C5_BP=1009)
for (label in names(sets)) {
  res <- read.delim(file.path(base,files[[label]]))
  sig <- res[is.finite(res$padj) & res$padj<0.05,]
  d <- sets[[label]]
  stopifnot(nrow(sig)==expected[[label]], !anyDuplicated(sig$pathway),
            all(sig$pathway %in% d$gs_name), all(d$db_version=='2026.1.Hs'))
  d <- unique(d[d$gs_name %in% sig$pathway & !is.na(d$ensembl_gene) & nzchar(d$ensembl_gene),
               c('gs_name','ensembl_gene','gene_symbol','gs_subcollection','db_version')])
  con <- gzfile(file.path(out,paste0(label,'_cached_memberships.tsv.gz')),'wt')
  write.table(d,con,sep='\t',row.names=FALSE,quote=FALSE,na='')
  close(con)
  cat(label,':',nrow(sig),'significant pathways;',nrow(d),'annotation rows\n')
}
