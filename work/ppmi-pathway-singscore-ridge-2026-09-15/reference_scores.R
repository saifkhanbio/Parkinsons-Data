# Independent base-R reference for upstream singleSingscore knownDirection=FALSE.
# Formula source: DavisLaboratory/singscore R/rankAndScoring.R, accessed 2026-09-15.
args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==3)
x<-as.matrix(read.delim(args[1],check.names=FALSE));m<-read.delim(args[2])
scores<-sapply(sort(unique(m$pathway)),function(p) {
  idx<-m$gene_index[m$pathway==p]+1L;k<-floor(length(idx)/2);B<-ceiling(ncol(x)/2)
  lower<-(k+1)/2;upper<-(2*B-k+1)/2
  apply(x,1,function(v) {r<-rank(v,ties.method='min');z<-abs(r-ceiling(median(r)));(mean(z[idx])-lower)/(upper-lower)})
})
write.table(scores,args[3],sep='\t',quote=FALSE,row.names=FALSE,col.names=FALSE)
