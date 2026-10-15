# Public release checks — October 15, 2026

These checks validate the public packaging and supported entry points. They
are separate from the historical performance experiments.

| Check | Result |
|---|---|
| Unit/provenance suite | 6 tests passed |
| Original-reference action comparison | 1,438 compared; 0 differences |
| Official native self-play, seed 0 | Both DONE; 76,801 / 76,801; both production matches |
| Official native self-play, seed 908391002 | Both DONE; 136,450 / 136,450; both production matches |
| Built archive in isolated Python and temporary directory | 720 frames; both DONE; 76,801 / 76,801; local-module paths all inside extracted archive |
| Crop-certificate regeneration | 173 source cycles; 112 field-compatible; 75 simpler-delivery candidates |
| Livestock-certificate regeneration | 14 placements; 7 eligible alternatives; source 192 milk / 170 wool |
| Contract compiler + full live game | 719 source transitions and 720 live frames; cash and inventory accounts passed; candidate production matched |
| Training-data preparation | One complete real reference replay; 6,264 feature rows |
| Portable training/export smoke | One original training episode and one original validation episode, 6,264 rows each; all four exported heads matched scikit-learn exactly |
| Credential/private-path scan | No findings in selected public text and decompressed assets |

The small training test checks the public training tool. Its newly fitted
model is **not** shipped in place of the original 200-episode model.
The final model file retains SHA-256
`d046ab2d3bf90df91a897ab9f5ac51481d73f91e572ac969acf9ab5b2c289600`.

The public inference tree contains the original dependency closure, not all
216 files of the historical archive. One certificate's three absolute
provenance path keys were redacted; policy fields and all inference Python
files remain unchanged. Public/original hashes and modifications are listed
in [source_manifest.json](source_manifest.json).

Raw reference replays, newly prepared training arrays, temporary extracted
archives and local logs are excluded from Git. Compact check outputs are in
[`results/release_checks.json`](../results/release_checks.json).
