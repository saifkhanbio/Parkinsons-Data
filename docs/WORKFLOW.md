# Analysis workflow

The source is organized by analysis stage. Intermediate matrices, metadata,
gene sets and fitted objects are private local prerequisites. Run dependent
stages only after reconstructing their inputs and reviewing their source paths.

## Ordered core stages

1. Clinical/release eligibility and broader cohort assembly: `ppmi-eligibility-audit` and `ppmi-broader-cohort`.
2. Raw-count extraction, expression QC and post-review cohort freeze: `ppmi-expression-qc`.
3. Covariate preparation and design-rank checks: `ppmi-model-design`.
4. Primary and alternative DESeq2 specifications; Hallmark and GO enrichment: `ppmi-deseq2`, including its `c2_c5bp` extension.
5. Immune-marker, covariate and influence review: `ppmi-results-review`, followed by `ppmi-qc-sensitivity`.
6. Gene annotation, prioritization, pathway membership and network construction: the annotation, prioritization, significant-pathway and Cytoscape modules.
7. Hematology matching and approved cohort assembly: `ppmi-blood-counts`; paired measured-cell models and collection-window checks: `ppmi-blood-cell-adjustment` and `ppmi-blood-cell-timing`.
8. Cell-adjusted candidate review: `ppmi-five-gene-review`.
9. Initial classifiers, repeated comparisons and revised solver: `ppmi-classifier`, `ppmi-classifier-refinement`, and `ppmi-classifier-refinement-retry`.
10. Hallmark score and member-gene representations, followed by controlled algorithm, regularization, PCA and gene-selection comparisons.
11. Feature-stability and two-gene covariate audits.

## Additional model-development modules

Dated demographic, class-weighting, random-forest, boosting, sex/age-stratified,
blood-block, pathway-score, intergenic-adjustment and training-calibration modules
extend the original workflow. Their inputs and source imports specify which
earlier baseline or assessment objects they reuse. Stratified DE modules learn
their selection within the training partitions defined by their own planning
scripts. They are distinct from the fixed Hallmark-member comparison.

## Source directory index

| Directory | Analysis source files |
| --- | --- |
| [work/ppmi-blood-cell-adjustment-2026-09-12](../work/ppmi-blood-cell-adjustment-2026-09-12/README.md) | `launch.py`, `run_models.R` |
| [work/ppmi-blood-cell-timing-2026-09-12](../work/ppmi-blood-cell-timing-2026-09-12/README.md) | `compare_windows.py`, `launch.py`, `run_models.R` |
| [work/ppmi-blood-counts-2026-09-12](../work/ppmi-blood-counts-2026-09-12/README.md) | `audit_blood_counts.py`, `freeze_approved_cohort.py` |
| [work/ppmi-broader-cohort-2026-09-12](../work/ppmi-broader-cohort-2026-09-12/README.md) | `define_ppmi_broader_cohort.py` |
| [work/ppmi-classifier-2026-09-12](../work/ppmi-classifier-2026-09-12/README.md) | `classifier.py`, `launch.py`, `predict.py` |
| [work/ppmi-classifier-feature-stability-2026-09-12](../work/ppmi-classifier-feature-stability-2026-09-12/README.md) | `review.py` |
| [work/ppmi-classifier-refinement-2026-09-12](../work/ppmi-classifier-refinement-2026-09-12/README.md) | `launch.py`, `predict.py`, `refine.py` |
| [work/ppmi-classifier-refinement-retry-2026-09-12](../work/ppmi-classifier-refinement-retry-2026-09-12/README.md) | `elastic_solver.py`, `launch.py`, `predict.py`, `refine.py`, `validate_solver_repair.py` |
| [work/ppmi-cytoscape-2026-09-12](../work/ppmi-cytoscape-2026-09-12/README.md) | `build_cytoscape.py`, `export_gene_sets.R`, `finish_cytoscape.py`, `prepare_network_analysis.py` |
| [work/ppmi-deseq2-2026-09-12](../work/ppmi-deseq2-2026-09-12/README.md) | `diagnose_influence.R`, `fetch_gene_sets.R`, `finish_analysis.py`, `gpu_postprocess.py`, `gpu_validate.py`, `prepare_inputs.py`, `run_deseq2.R`, `run_enrichment.R`, `summarize_results.py` |
| [work/ppmi-deseq2-2026-09-12/c2_c5bp](../work/ppmi-deseq2-2026-09-12/c2_c5bp/README.md) | `finalize_gpu.py`, `run_enrichment.R` |
| [work/ppmi-eligibility-audit/ir3-audit-2026-09-12](../work/ppmi-eligibility-audit/ir3-audit-2026-09-12/README.md) | `audit_ppmi_ir3.py` |
| [work/ppmi-expression-qc-2026-09-12](../work/ppmi-expression-qc-2026-09-12/README.md) | `extract_ppmi_selected_counts.py`, `finalize_ppmi_qc.py`, `qc_ppmi_after_review.R`, `qc_ppmi_counts.R` |
| [work/ppmi-five-gene-review-2026-09-12](../work/ppmi-five-gene-review-2026-09-12/README.md) | `review.py`, `review_influence.R` |
| [work/ppmi-gene-annotation-review-2026-09-12](../work/ppmi-gene-annotation-review-2026-09-12/README.md) | `review_annotations.py` |
| [work/ppmi-gene-prioritization-2026-09-12](../work/ppmi-gene-prioritization-2026-09-12/README.md) | `prioritize_genes.py` |
| [work/ppmi-hallmark-blood-block-ridge-2026-09-15](../work/ppmi-hallmark-blood-block-ridge-2026-09-15/README.md) | `blood_models.py`, `launch.py`, `predict.py`, `reporting.py`, `worker.py` |
| [work/ppmi-hallmark-classifier-2026-09-12](../work/ppmi-hallmark-classifier-2026-09-12/README.md) | `export_hallmark.R`, `launch.py`, `pathways.py`, `predict.py` |
| [work/ppmi-hallmark-pathway-catboost-2026-09-14](../work/ppmi-hallmark-pathway-catboost-2026-09-14/README.md) | `predict.py`, `run.py` |
| [work/ppmi-mapped-gene-boosting-2026-09-14](../work/ppmi-mapped-gene-boosting-2026-09-14/README.md) | `estimators.py`, `launch.py`, `predict.py` |
| [work/ppmi-mapped-gene-classifier-2026-09-12](../work/ppmi-mapped-gene-classifier-2026-09-12/README.md) | `launch.py`, `mapped_genes.py`, `predict.py` |
| [work/ppmi-mapped-gene-model-comparison-2026-09-12](../work/ppmi-mapped-gene-model-comparison-2026-09-12/README.md) | `estimators.py`, `launch.py`, `predict.py` |
| [work/ppmi-mapped-gene-selection-2026-09-12](../work/ppmi-mapped-gene-selection-2026-09-12/README.md) | `gene_selection.py`, `launch.py`, `predict.py` |
| [work/ppmi-mapped-gene-svm-2026-09-12](../work/ppmi-mapped-gene-svm-2026-09-12/README.md) | `launch.py`, `mapped_svm.py`, `predict.py` |
| [work/ppmi-mapped-pca-ridge-2026-09-12](../work/ppmi-mapped-pca-ridge-2026-09-12/README.md) | `launch.py`, `pca_ridge.py`, `predict.py` |
| [work/ppmi-mapped-ridge-tuning-2026-09-12](../work/ppmi-mapped-ridge-tuning-2026-09-12/README.md) | `launch.py`, `predict.py`, `ridge_tuning.py` |
| [work/ppmi-mapped-top100-ridge-2026-09-12](../work/ppmi-mapped-top100-ridge-2026-09-12/README.md) | `launch.py`, `predict.py`, `top100.py` |
| [work/ppmi-model-design-2026-09-12](../work/ppmi-model-design-2026-09-12/README.md) | `check_ppmi_design.R`, `prepare_ppmi_covariates.py` |
| [work/ppmi-pathway-singscore-ridge-2026-09-15](../work/ppmi-pathway-singscore-ridge-2026-09-15/README.md) | `export_sets.R`, `launch.py`, `path_models.py`, `predict.py`, `prepare_mapping.py`, `reference_scores.R`, `reporting.py`, `validation.py`, `worker.py` |
| [work/ppmi-qc-sensitivity-2026-09-12](../work/ppmi-qc-sensitivity-2026-09-12/README.md) | `finalize_gpu.py`, `run_pipeline.sh`, `run_sensitivity.R` |
| [work/ppmi-random-forest-comparison-2026-09-14](../work/ppmi-random-forest-comparison-2026-09-14/README.md) | `forest.py`, `launch.py`, `report.py` |
| [work/ppmi-results-review-2026-09-12](../work/ppmi-results-review-2026-09-12/README.md) | `cell_scores_gpu.py`, `export_pathways.R`, `finalize_review.py`, `prepare_confounders.py`, `review_influence.R`, `review_models.R` |
| [work/ppmi-results-review-2026-09-12/site_race_pathways](../work/ppmi-results-review-2026-09-12/site_race_pathways/README.md) | `finalize_gpu.py`, `run_enrichment.R` |
| [work/ppmi-ridge-class-weighting-2026-09-14](../work/ppmi-ridge-class-weighting-2026-09-14/README.md) | `launch.py`, `report.py`, `weighting.py` |
| [work/ppmi-ridge-demographic-audit-2026-09-14](../work/ppmi-ridge-demographic-audit-2026-09-14/README.md) | `run_audit.py` |
| [work/ppmi-ridge-demographic-models-2026-09-14](../work/ppmi-ridge-demographic-models-2026-09-14/README.md) | `demographic_models.py`, `launch.py`, `predict.py`, `reporting.py` |
| [work/ppmi-ridge-error-audit-2026-09-15](../work/ppmi-ridge-error-audit-2026-09-15/README.md) | `audit.py` |
| [work/ppmi-ridge-intergenic-adjustment-2026-09-15](../work/ppmi-ridge-intergenic-adjustment-2026-09-15/README.md) | `adjustment.py`, `launch.py`, `predict.py`, `reporting.py`, `worker.py` |
| [work/ppmi-ridge-training-calibration-2026-09-15](../work/ppmi-ridge-training-calibration-2026-09-15/README.md) | `apply_calibrator.py`, `calibrator.py`, `launch.py`, `worker.py` |
| [work/ppmi-sex-age-stratified-de-ridge-2026-09-14](../work/ppmi-sex-age-stratified-de-ridge-2026-09-14/README.md) | `de_workers.py`, `launch.py`, `model_utils.py`, `prepare_plan.py`, `reporting.py`, `run_selector.R`, `weighted_metrics.py` |
| [work/ppmi-sex-age-stratified-rf-2026-09-14](../work/ppmi-sex-age-stratified-rf-2026-09-14/README.md) | `forest.py`, `launch.py`, `model_utils.py`, `plan_checks.py`, `reporting.py`, `weighted_metrics.py` |
| [work/ppmi-sex-pooled-fdr-only-ridge-2026-09-14](../work/ppmi-sex-pooled-fdr-only-ridge-2026-09-14/README.md) | `launch.py`, `model_utils.py`, `plan_checks.py`, `reporting.py`, `weighted_metrics.py`, `worker.py` |
| [work/ppmi-significant-pathway-genes-2026-09-13](../work/ppmi-significant-pathway-genes-2026-09-13/README.md) | `export_memberships.R`, `extract_genes.py` |
| [work/ppmi-svm-learning-curves-2026-09-12](../work/ppmi-svm-learning-curves-2026-09-12/README.md) | `launch.py`, `predict.py`, `svm_analysis.py` |
| [work/ppmi-two-gene-audit-2026-09-12](../work/ppmi-two-gene-audit-2026-09-12/README.md) | `audit.py` |
