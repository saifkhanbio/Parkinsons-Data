args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1)
root<-normalizePath(args[1]);cache<-file.path(dirname(root),'ppmi-deseq2-2026-09-12','cache','R','msigdbr')
for (collection in c('H','C2','C5')) {
  d<-readRDS(file.path(cache,paste0('msigdb.2026.1.Hs.',collection,'.rds')))
  stopifnot(is.data.frame(d),all(d$db_version=='2026.1.Hs'),all(d$gs_collection==collection))
  if(collection=='H') label<-'Hallmark'
  if(collection=='C2') {d<-d[d$gs_subcollection=='CP'|startsWith(d$gs_subcollection,'CP:'),];label<-'C2_CP'}
  if(collection=='C5') {d<-d[d$gs_subcollection=='GO:BP',];label<-'C5_BP'}
  d<-unique(d[,c('gs_id','gs_name','gs_collection','gs_subcollection','gs_description','db_ensembl_gene','db_gene_symbol','db_version')])
  con<-gzfile(file.path(root,paste0(label,'_annotation.tsv.gz')),'wt');write.table(d,con,sep='\t',quote=TRUE,row.names=FALSE,na='');close(con)
  cat(label,':',length(unique(d$gs_name)),'sets;',nrow(d),'annotation rows\n')
}
