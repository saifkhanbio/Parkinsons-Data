suppressPackageStartupMessages(library(edgeR))
out <- 'outputs/ppmi-expression-qc-2026-09-12'
c <- as.matrix(read.delim(gzfile(file.path(out,'raw_counts.tsv.gz')),row.names=1,check.names=FALSE))
d <- read.delim(file.path(out,'QC_disposition.tsv'),stringsAsFactors=FALSE)
k <- d$excluded_independent_QC=='False'
stopifnot(sum(k)==558)
c <- c[,as.character(d$PATNO[k]),drop=FALSE]; d<-d[k,]
y<-DGEList(c,group=d$group); keep<-filterByExpr(y,group=d$group)
y<-calcNormFactors(y[keep,,keep.lib.sizes=FALSE]); e<-cpm(y,log=TRUE,prior.count=2)
p<-prcomp(t(e),rank.=5);v<-p$sdev^2/sum(p$sdev^2)*100
saveRDS(list(raw_counts=c,metadata=d,expression_filter=keep,exploratory_logCPM=e),file.path(out,'cohort_after_QC.rds'))
png(file.path(out,'PCA_after_QC.png'),width=1800,height=650,res=150)
par(mfrow=c(1,3),mar=c(4,4,3,1))
for (field in c('group','phase','RIN')) {
 x<-d[[field]]
 if(field=='RIN') { cols<-colorRampPalette(c('#9A3B26','#ECD68A','#277A8C'))(100)[pmax(1,pmin(100,round(x*10)))]; title<-'RNA integrity (red low, blue high)' }
 else { f<-factor(x);cols<-c('#287F9C','#D86738')[f];title<-if(field=='group') 'Diagnosis group' else 'Sequencing phase' }
 plot(p$x[,1],p$x[,2],col=cols,pch=19,cex=.55,xlab=sprintf('PC1 (%.1f%%)',v[1]),ylab=sprintf('PC2 (%.1f%%)',v[2]),main=title)
 if(field!='RIN')legend('topright',levels(f),col=c('#287F9C','#D86738'),pch=19,bty='n',cex=.8)
}
dev.off()
write.table(data.frame(PATNO=d$PATNO,p$x),file.path(out,'PCA_after_QC.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
cat('Retained:',ncol(c),'Expression-filtered genes:',sum(keep),'PC variance:',v[1:5],'\n')
