args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2, !file.exists(args[2]))
x <- readRDS(args[1])
h <- x[x$gs_collection=="H", c("gs_name","ensembl_gene","gene_symbol","db_version","gs_description","gs_url")]
stopifnot(length(unique(h$gs_name))==50, all(h$db_version=="2026.1.Hs"))
write.table(h,args[2],sep="\t",row.names=FALSE,quote=FALSE,na="")
cat("Exported",nrow(h),"membership rows from",length(unique(h$gs_name)),"Hallmark sets\n")
