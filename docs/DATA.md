# Data and learned-model provenance

## Included assets

| Asset | Purpose | Origin |
|---|---|---|
| `agent/asset.json.gz` | Compiled production schedule and reference farms | Daily Top Episodes, 2026-09-26, episode 113905606, seat 0 |
| `agent/market_game_20260930/g4/cycle_bank.json` | Short crop-cycle compatibility data | Project analysis of the production schedule |
| `agent/market_game_20260930/g8/pasture_certificates_v2.json` | Complete livestock lifetimes and compatibility | Project simulation under official rules |
| `agent/market_game_20260930/g2/opponent_flow_v1.json` | Portable trained sale-volume predictor | 200 training episodes; 99 separate validation episodes |

The runtime files originate from submission 56707930. The
[source manifest](../provenance/source_manifest.json) records their original
workspace-relative source and SHA-256. Three private absolute path keys in
certificate provenance were replaced by portable labels; certificate contents
used by the policy are unchanged. Original and public hashes are both kept.

## Obtaining replays

Start at Kaggle's [Daily Top Episodes discussion](https://www.kaggle.com/competitions/kaggriculture/discussion/731215)
and follow its current dataset/download instructions while signed in to your
own account. The repository does not include credentials or assume that a
temporary signed URL is permanent. Download the relevant complete episode
JSON files and, if necessary, decompress the daily archives locally.

The replay parser accepts `.json` and `.json.gz` with `configuration` and
720 `steps`, including both seats' complete observations and actions.
Incomplete or version-mismatched data should be rejected, not silently
converted into successful labels. Data must be compatible with engine 1.32.7.

Put local downloads under ignored `data/`. Do not commit authentication files
or raw collections. See [attribution and source terms](../THIRD_PARTY_NOTICES.md).

## Replay manifest

Create `data/replays.json`. Paths are relative to that manifest. IDs must be
numeric and unique across splits. Use separate episodes and preferably later
dates for validation, not random rows from the same episode.

```json
{
  "rows": [
    {"episode_id":"12345", "date":"2026-09-01", "split":"train", "replay":"replays/12345.json.gz"},
    {"episode_id":"12346", "date":"2026-09-10", "split":"validation", "replay":"replays/12346.json.gz"}
  ]
}
```

The example IDs are placeholders. The original actual episode list, dates,
split and feature-file hashes are in
[`provenance/training_split.json`](../provenance/training_split.json).

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_flow_data.py --replays data/replays.json --out-dir data/flow
python scripts/train_flow.py --manifest data/flow/data_manifest.json --out-dir results/local/retrained
```

The preparation command uses the official clearing rules to count **successful
SELL fills**, not the requested order quantity. Source action `t+1` belongs to
transition `t -> t+1`. It builds public features with the same online tracker
used by inference, while future fills appear only in offline labels.

Rows use steps 0–695, both seats, stride 2 with staggered phase. All nine
products share one feature representation; there is no product ID, episode
ID, private rival warehouse or unrevealed shop sequence in the input.

## Final predictor

- Training: 200 episodes, August 15–September 8, 2026; 1,252,800 rows.
- Validation: 99 episodes, September 9–16; 620,136 rows.
- Inputs: 16 public features covering time, inventory/price, known demand,
  visible rival production and causal public-flow history.
- Targets: sale volume in `[0,1)`, `[1,4)`, `[4,12)`, `[12,24)` future turns.
- Four `HistGradientBoostingRegressor` heads: Poisson loss, 80 iterations,
  at most 15 leaves, depth 4, minimum leaf size 200, L2 10, learning rate 0.08,
  no early stopping, seeds 908330000–908330003.
- Portable tree predictions checked against scikit-learn on up to 1,500
  validation rows per head; maximum allowed error `1e-7`.

The public trainer writes to an explicit new output directory. It does not
overwrite the frozen released model. Original training entry points are
retained under `research/original_training`; the portable scripts adapt path
handling and data loading, not the fitting objective or hyperparameters.

Model SHA-256:

```text
d046ab2d3bf90df91a897ab9f5ac51481d73f91e572ac969acf9ab5b2c289600
```

The model estimates conditional average volume. Rare bursts, changing rival
responses and newer playing styles are not fully represented by a mean
forecast. Judge a retrained model through full-game cash difference as well as
prediction errors. Original error metrics are in
[`results/historical/flow_training.json`](../results/historical/flow_training.json).
