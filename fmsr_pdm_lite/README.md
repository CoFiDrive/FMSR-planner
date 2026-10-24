# FMSR PDM-Lite adapter

`fmsrpdm/` contains the planner, training, inference, and evaluation adapters.
`scripts/` contains the launchers and `tests/` contains their regression checks.
`vendor/` contains the frozen perception dependency; `upstream/` contains original
reference snapshots and provenance metadata.

Supply external perception weights, anchor banks, normalization statistics,
datasets and CARLA resources before running the relevant launcher. Set resource
paths for your own environment. Training and evaluation outputs are excluded.

The original reference files now live under `upstream/planner_reference/`.
Original attribution and source hashes are preserved in `THIRD_PARTY_NOTICES.md`
and `upstream/sources.json`; their historical source paths describe the upstream
repository. See the repository-root `THIRD_PARTY_NOTICES.md` for the naming map.
