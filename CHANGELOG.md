# Changelog

This project follows [Semantic Versioning](https://semver.org/). Because the pipeline is driven by
Hydra configs and trained checkpoints, a **MAJOR** bump means a config key, output schema, or
default-checkpoint change that breaks existing runs; **MINOR** adds capability without breaking them;
**PATCH** is fixes only. Job-level history lives in `JOB_LEDGER.md`.

## 1.0.0 - 2026-11-04

First tagged release.

**Default models** (see `boem_conf/`)
- Detector: `1968e962/epoch29` (single class `Object`, accepted score floor `predict.min_score=0.70`).
  Published to Hugging Face as `weecology/deepforest-marine-biodiversity` (PR #3).
- Classifier: flat `a3dc30a0`, `use_metadata=false`, plus the hierarchical H-CAST head.

**Pipeline**
- Active-learning / full-flight prediction -> Label Studio review -> annotation download, with every
  annotation CSV mirrored in git under `annotations_backup/`.
- Reports regenerate from the prediction cache and annotations without a GPU (`report_only=True`).
- Review output carries an explicit `human_reviewed` column and `final_label`; the old `score=2.0`
  sentinel is gone. A box rejected in Label Studio is `FalsePositive`.
- Flights with fewer than 25 detected frames are reviewed in full.

**Evaluation and training data**
- Zero-shot holdout is pinned to `JPG_20260202_141900` + `JPG_20260201_134000` and dropped by
  flight column in `scripts/prepare_USGS.py`. Earlier splits leaked these flights.
- Family-rank dolphin boxes are dropped from the flat classifier (kept in the hierarchical one);
  dolphin species review folded into classifier training via IoU overlay.
- Classifier class-balance cap is configurable.

**Known limits**
- Flat-classifier zero-shot scoring of the latest retrain (49329504) is not done.
- No trustworthy in-distribution box recall exists for the production detector; report zero-shot
  and validation recall separately.
- Upgraded-camera imagery (13400x9528) is untested beyond a single frame.
