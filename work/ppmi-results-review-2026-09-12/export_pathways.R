script <- sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1])
out <- dirname(normalizePath(script)); a<-file.path(dirname(out),"ppmi-deseq2-2026-09-12")
s<-readRDS(file.path(a,"msigdb_hallmark_gobp.rds"))
s<-s[s$gs_collection=="H",c("gs_name","ensembl_gene")]
write.table(unique(s),file.path(out,"hallmark_membership.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
