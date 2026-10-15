# Attribution and license scope

Original code and documentation in this repository are released under the
[MIT License](LICENSE). This does not relicense third-party materials or grant
rights to Kaggle competition data, names, or trademarks.

## Kaggle game engine

`evaluation/scenario_engine.py` is a modified copy of the Kaggriculture engine
distributed with **kaggle-environments 1.32.7**, from the
[Kaggle/kaggle-environments project](https://github.com/Kaggle/kaggle-environments).
It remains under the [Apache License 2.0](licenses/Apache-2.0.txt).
The modification loads the installed specification and allows an offline
evaluator to supply the recorded shop-opening sequence. The deployed agent
uses the installed official engine, not this scenario engine.

Runtime code also calls official pricing, production and settlement helpers.
The installed engine, PyTorch, NumPy, scikit-learn, orjson and threadpoolctl
retain their respective upstream licenses; their distributions are not
vendored here.

## Replay-derived production assets

The production schedule in `agent/asset.json.gz` is a compiled game-action
and farm-state representation derived from public Daily Top Episodes data:
source **2026-09-26 / episode 113905606 / seat 0**. The associated crop cycle
bank and livestock certificates are project-generated derivatives of that
schedule. The original player's demonstrated production route is credited
as a source, not represented as a route invented from scratch by this project.

Source collection: [Daily Top Episodes](https://www.kaggle.com/competitions/kaggriculture/discussion/731215).
The raw collection is not redistributed. Public availability is not a grant
of an MIT license over replay data. Use and redistribution of source data
remain subject to the applicable [competition rules](https://www.kaggle.com/competitions/kaggriculture/rules)
and source terms. MIT covers our implementation and original contributions,
not rights held by the competition or other participants.

The exported opponent-flow model was trained by this project on public
replays. Its episode IDs, feature names and training provenance are recorded
in the exported JSON and [data documentation](docs/DATA.md).

## Related solutions and documentation references

The report cites other teams' approaches, including
[M & M & P & Q](https://github.com/msdsm/kaggriculture-solution) and
[Decem-Y's behavior-cloning solution](https://github.com/Decem-Y/kaggriculture-bc).
Their repository organization informed the separation of inference, training,
evaluation and provenance here. No code from those repositories is bundled.
