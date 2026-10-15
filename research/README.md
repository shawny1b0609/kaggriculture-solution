# Selected historical implementations

These original files document the final solution's lineage:

- `original_training/build_data.py`: original public-feature data extraction.
- `original_training/train.py`: fitting and portable-tree export.
- `original_certification/certify.py`: livestock lifetime certification.
- `original_evaluation/fixed_supply_probe.py`: production-contract compilation,
  debt/cost accounting and the original experiment runner.

They retain historical workspace-relative loaders and are reference source,
not the supported command-line interface. Use the corresponding portable
commands under `scripts/`. Those adapters preserve the underlying fitting,
certification and accounting logic while removing dependencies on the full
private research directory.

The earlier route-forest and RL explorations are described in
[`docs/RESEARCH_HISTORY.md`](../docs/RESEARCH_HISTORY.md) and the full report.
Their experimental checkpoints are not deployed by the final solution and
are not required to run it.
