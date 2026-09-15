suppressPackageStartupMessages(library(edgeR))
suppressPackageStartupMessages(library(jsonlite))
out <- 'outputs/ppmi-expression-qc-2026-09-12'
counts <- as.matrix(read.delim(gzfile(file.path(out,'raw_counts.tsv.gz')), row.names=1, check.names=FALSE))
meta <- read.delim(file.path(out,'sample_metadata.tsv'), check.names=FALSE, stringsAsFactors=FALSE)
if (!'Sample' %in% names(meta)) {
  manifest <- fromJSON('outputs/ppmi-broader-cohort-2026-09-12/cohort_manifest.json')
  meta$Sample <- manifest$sample[match(as.character(meta$PATNO),manifest$PATNO)]
  write.table(meta,file.path(out,'sample_metadata.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
}
stopifnot(identical(colnames(counts),as.character(meta$PATNO)),ncol(counts)==579,all(is.finite(counts)),all(counts>=0),all(counts==floor(counts)))
lib <- colSums(counts);stopifnot(all(lib>0))
y <- DGEList(counts=counts,group=meta$group)
keep <- filterByExpr(y, group=meta$group)
y <- calcNormFactors(y[keep,,keep.lib.sizes=FALSE],method='TMM')
lcpm <- cpm(y,log=TRUE,prior.count=2)
pc <- prcomp(t(lcpm),center=TRUE,scale.=FALSE,rank.=10)
variance <- pc$sdev^2/sum(pc$sdev^2)*100
rz <- function(x) { s<-mad(x,na.rm=TRUE); if (!is.finite(s)||s==0) return(rep(0,length(x))); (x-median(x,na.rm=TRUE))/s }
num <- function(x) suppressWarnings(as.numeric(x))
qc <- data.frame(PATNO=meta$PATNO,group=meta$group,assigned_counts=lib,genes_nonzero=colSums(counts>0),genes_CPM1=colSums(cpm(counts)>=1),top_gene_fraction=apply(counts,2,max)/lib,TMM_factor=y$samples$norm.factors,RIN=num(meta[['RIN Value']]),uniquely_mapped_percent=num(meta$uniquely_mapped_percent),phase=sub('^(PPMI-Phase[12]).*','\\1',meta$Sample),sex=meta$GENDER)
qc$low_library <- rz(log10(qc$assigned_counts)) < -3
qc$low_detection <- rz(qc$genes_CPM1) < -3
qc$low_RIN <- !is.na(qc$RIN)&qc$RIN<6
qc$low_mapping <- !is.na(qc$uniquely_mapped_percent)&qc$uniquely_mapped_percent<70
pcz <- apply(pc$x[,1:5,drop=FALSE],2,rz)
qc$extreme_PC <- apply(abs(pcz)>6,1,any)
qc$review_flag <- with(qc,low_library|low_detection|low_RIN|low_mapping|extreme_PC)
qc <- cbind(qc,pc$x)
corr <- cor(lcpm)
diag(corr)<-NA
qc$median_sample_correlation <- apply(corr,2,median,na.rm=TRUE)
near <- which(corr>0.995 & upper.tri(corr),arr.ind=TRUE)
pairs <- if(nrow(near)) data.frame(PATNO1=colnames(corr)[near[,1]],PATNO2=colnames(corr)[near[,2]],correlation=corr[near]) else data.frame(PATNO1=character(),PATNO2=character(),correlation=numeric())
write.table(qc,file.path(out,'sample_QC.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
write.table(pairs,file.path(out,'high_correlation_pairs.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
write.table(data.frame(Geneid=rownames(counts),expression_filter_pass=keep),file.path(out,'gene_filter.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
saveRDS(list(logCPM=lcpm,PCA=pc,metadata=meta,QC=qc),file.path(out,'exploratory_QC.rds'))
colors <- ifelse(meta$group=='PD','#D86738','#287F9C')
png(file.path(out,'QC_overview.png'),width=1800,height=1400,res=160)
par(mfrow=c(2,2),mar=c(4.5,4.5,3,1))
plot(pc$x[,1],pc$x[,2],col=colors,pch=19,cex=.65,xlab=sprintf('PC1 (%.1f%%)',variance[1]),ylab=sprintf('PC2 (%.1f%%)',variance[2]),main='Whole-blood expression: all 579 samples')
legend('topright',c('PD','Control'),col=c('#D86738','#287F9C'),pch=19,bty='n')
plot(log10(lib),qc$genes_CPM1,col=colors,pch=19,cex=.65,xlab='log10 assigned counts',ylab='Genes with CPM >= 1',main='Library size and gene detection')
boxplot(RIN~group,data=qc,col=c('#A7D2DE','#EAB298'),ylab='RNA integrity number',main='RNA quality from release metadata')
plot(qc$uniquely_mapped_percent,qc$median_sample_correlation,col=colors,pch=19,cex=.65,xlab='Uniquely mapped reads (%)',ylab='Median sample correlation',main='Mapping and expression consistency')
dev.off()
summary <- list(stage='QC_METRICS_COMPLETE_REVIEW_PENDING',samples=ncol(counts),genes_imported=nrow(counts),genes_expression_filter=sum(keep),group_counts=as.list(table(meta$group)),flag_counts=as.list(colSums(qc[,c('low_library','low_detection','low_RIN','low_mapping','extreme_PC','review_flag')])),flagged_PATNO=qc$PATNO[qc$review_flag],PC_variance_percent=variance[1:5],phase_by_group=table(qc$phase,qc$group),high_correlation_pairs=nrow(pairs),assigned_count_range=range(lib),RIN_range=range(qc$RIN,na.rm=TRUE),mapping_range=range(qc$uniquely_mapped_percent,na.rm=TRUE))
summary$phase_by_group <- as.data.frame(summary$phase_by_group)
write_json(summary,file.path(out,'QC_summary.json'),pretty=TRUE,auto_unbox=TRUE)
capture.output(sessionInfo(),file=file.path(out,'R_session_info.txt'))
print(summary)
