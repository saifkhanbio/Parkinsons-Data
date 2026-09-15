# cytoscape 2026 09 12

Public analysis-source directory. Generated data, result summaries, models,
figures and the original result-bearing README are not part of this upload.

See the [workflow](../../docs/WORKFLOW.md),
[execution guide](../../docs/REPRODUCIBILITY.md) and
[private-input requirements](../../docs/DATA_ACCESS.md).

## Source files

- [build_cytoscape.py](build_cytoscape.py): Create styled gene networks and an EnrichmentMap; save a Cytoscape session.
- [export_gene_sets.R](export_gene_sets.R)
- [finish_cytoscape.py](finish_cytoscape.py): Validate native EnrichmentMap edges, annotate effects, and export the session.
- [prepare_network_analysis.py](prepare_network_analysis.py): Validate STRING mappings; calculate network metrics and pathway overlaps.

## Execution notes

Run from the repository root unless the script explicitly resolves its own
directory. Read the input/output path constants before execution. Inputs from
earlier stages must be generated or restored from authorized local records.
Study-specific checks, seeds and model settings remain in the source; keep
participant data and all generated artifacts outside version control.
