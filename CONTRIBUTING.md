# Contributing

Please describe the observed behavior, expected behavior, engine version,
seed, and a minimal reproduction command. Do not attach credentials or
nonpublic replays.

Run `python -m unittest discover -s tests -v` before proposing a change.
Changes to farm execution should additionally run full native games and
compare production with the selected plan. Market changes should report
both cash and cash difference against the same opponents in both seats.

The files under `agent/` preserve the submitted implementation. If you change
them, clearly label the result as a new candidate, retain the original
provenance, and update tests that assert frozen-source identity. Do not silently
rewrite historical result tables to describe a new version.
