script <- sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])
out <- dirname(normalizePath(script))
.libPaths(c(file.path(out, "R-library"), .libPaths()))
Sys.setenv(R_USER_CACHE_DIR=file.path(out, "cache"))
suppressPackageStartupMessages(library(msigdbr))
suppressPackageStartupMessages(library(jsonlite))
hallmark <- msigdbr(db_species="HS", species="Homo sapiens", collection="H")
go <- msigdbr(db_species="HS", species="Homo sapiens", collection="C5", subcollection="GO:BP")
sets <- rbind(hallmark, go)
saveRDS(sets, file.path(out, "msigdb_hallmark_gobp.rds"))
mapping <- unique(sets[,c("ensembl_gene", "gene_symbol", "ncbi_gene")])
write.table(mapping, file.path(out, "msigdb_gene_mapping.tsv"), sep="\t", quote=FALSE, row.names=FALSE)
write_json(list(msigdbr_version=as.character(packageVersion("msigdbr")),
                db_version=unique(sets$db_version), species="Homo sapiens",
                hallmark_sets=length(unique(hallmark$gs_name)),
                gobp_sets=length(unique(go$gs_name)),
                retrieval_time=format(Sys.time(), tz="UTC", usetz=TRUE),
                source="https://zenodo.org/records/18968178",
                documentation="https://igordot.github.io/msigdbr/articles/msigdbr-intro.html"),
           file.path(out, "gene_set_provenance.json"), pretty=TRUE, auto_unbox=TRUE)
cat("Saved", length(unique(hallmark$gs_name)), "Hallmark and", length(unique(go$gs_name)), "GO:BP sets\n")
