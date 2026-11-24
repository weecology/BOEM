# Detector architecture comparison: ResNet-50 RetinaNet vs DINOv3 RetinaNet vs DETR

Branch `detector-architecture-comparison`, worktree
`/blue/ewhite/b.weinstein/src/BOEM/.claude/worktrees/detector-architecture-comparison`.
Started 2026-11-23.

## Question
Is a next-generation detector (DINOv3 ViT backbone, or a transformer DETR head) worth
moving production off the ResNet-50 RetinaNet, given that DeepForest will maintain both?

## Environment (this branch only)
- DeepForest pinned to weecology/DeepForest PR #1473 (`jveitchmichaelis` commit `ac213f01`, draft):
  adds `architecture: retinanet_dinov3`, sitting on current DeepForest main, which already has
  HF `AutoModelForObjectDetection` (`architecture: DeformableDetr` / `automodel`).
- This venv **drops** friendly-beaver's CropModel metadata support (PR #1334). Use it for detection
  training/eval only. Production cannot adopt a new detector until #1473 and #1334 are both on one
  DeepForest ref.
- Hugging Face: `export HF_HOME=/blue/ewhite/b.weinstein/.cache/huggingface` (shared, already holds the
  gated-model token and `facebook/dinov3-vitl16-pretrain-sat493m`). No token is copied into this repo.
- Versions: deepforest 2.1.0 @ac213f01, transformers 5.9.0, torch 2.11.0+cu128, Python 3.11.

## Verified 2026-11-23 (CPU, one real 1000 px tile)
| arm | class | params | builds + forwards |
|---|---|---|---|
| retinanet (R50) | RetinaNetHub | 32M | yes |
| retinanet_dinov3, sat493m ViT-L/16 | RetinaNetDinov3Hub | 319M | yes |
| DeformableDetr, SenseTime/deformable-detr (COCO) | AutomodelWrapper | 40M | yes, 300 queries |

**Every arm, production included, resizes 1000 px tiles to 800 px** (torchvision `min_size=800`; DETR
processor `shortest_edge=800`). The validation set's median box is 27.6 px (61% < 32 px, 16% < 16 px),
so after the resize the median bird is ~22 px against a smallest anchor of 32 px. A stride-16 ViT is
hit hardest by this. Input size must be held fixed across arms in Phase 1 and tested on its own in Phase 2.
Densest validation tile has 158 boxes, so DETR's 300 queries do not cap.

## Fixed protocol (same as YOLO26 / c6ecec12 comparisons)
- Data: the 11-20 split in `/blue/ewhite/b.weinstein/BOEM/training/crops/`: train `train_max_empty_0_10.csv`,
  monitor `test_max_empty_0_25.csv`. Reference = production **c6ecec12/epoch27** (trained on exactly this
  split), so it is not retrained.
- **VALIDATION** (in-distribution): `scripts/eval_detection_checkpoint.py`, pooled TP/GT, IoU 0.4.
- **ZERO-SHOT** (pinned holdout JPG_20260202_141900 + JPG_20260201_134000): `scripts/threshold_sweep.py`.
- Report the two as separate, labelled numbers. Compare at **matched precision**, at the recall
  ceiling, and at the production gate **0.60** (moved from 0.70 on 2026-11-23 with c6ecec12), not only at a fixed threshold (score calibration differs between architectures).
- Also: recall by box size (<16, 16-32, >32 px) and inference cost (frames/s on a full flight, since surveys are ~500k frames).
- **Input size changes validation and zero-shot differently** (dense-flock chat, 52502538 + 52502730, c6ecec12 inference
  only): validation recall at equal precision 0.97 rises +7 / +11 at 1000 / 1200 px, but zero-shot at ~85.4% precision is
  flat (800 59.0 / 1000 ~57 / 1200 60.5), because holdout objects are big (median 71 px) and USGS validation birds small
  (~28 px). July review load at 0.60 per 5,000 frames: 17 / 12 / 28. So an architecture gain on small USGS birds may not
  transfer to BOEM survey imagery: always report validation and zero-shot separately, and add box-size bins to both.
- **Equal precision is interpolated** between grid thresholds, not density_strata.py's 0.05-grid pick (mis-buckets
  near the target; fixed in 9bd5e54).
- **Do not use `flock_dataset/` as a dense-bird set** without dropping `label == "FalsePositive"`: its non-training test
  frames hold 1,675 FalsePositive boxes vs 79 birds. Feb 2022 cap-probe frames are clean (`Object`) but mostly training frames.
- Underfitting test: focal cls loss on TRAIN tiles rises 0.050 (singles) -> 0.168 (21-50 birds), same as val
  (`/blue/ewhite/b.weinstein/BOEM/high_loss_tiles/tile_losses.csv`). Score each candidate on its own dense training tiles.
- **Dense flocks** (agreed with the dense-flock chat, 2026-11-23): score raw predictions (score_thresh <= 0.3) with
  `scripts/density_strata.py <predictions.csv> [gt.csv]` (from dense-flock-diagnostics commit 9bd5e54, interpolated equal precision + @0.60) on the validation
  set AND on the training-tile sample `/blue/ewhite/b.weinstein/BOEM/dense_flock/phase0/0d_train_sample.csv`.
  Reports @0.40/@0.70 and at matched overall precision 0.978. Baseline c6ecec12 @0.70 train/val recall: 29/30% at
  11-20 birds/tile, 24/27% at 21-50 (vs 61% on 1-bird tiles), i.e. R50 underfits flock birds even on its own training
  tiles. Whether DINOv3/DETR can fit them is a sharp test. Do NOT use Feb 2022 cap-probe frames or flock_dataset test
  as held-out dense sets (mostly training frames). A double-annotated dense held-out set may come from that chat.

## Phases
0. **Plumbing** (no science). Add `--architecture` (+ backbone/freeze/precision) to
   `scripts/USGS_backbone.py`; 1-GPU smoke train of each arm (a few batches). Confirm checkpoints reload
   in both eval scripts. Re-score c6ecec12 under this venv: it must reproduce 45.7 / 97.9 validation and
   59.0 / 85.4 zero-shot at 0.70, or the DeepForest version drift is a confound.
1. **Architecture at fixed input, native 1000 px (`min_size`/`max_size` raised so tiles are not shrunk), same split,
   same augmentations.** The 800 px default costs c6ecec12 ~7 pts validation recall at equal precision 0.97 by inference
   alone (dense-flock 0c: 55.4 -> 62.4 native -> 66.7 at 1200). The R50 reference is c6ecec12 predicted at 1000 (and 1200).
   - B: RetinaNet + DINOv3 ViT-L sat493m, encoder unfrozen (MillionTrees E007 recipe: SGD, cosine, bf16,
     grad clip 1.0). E007 gain over the fine-tuned R50: macro F1 +0.8 within-distribution, +4.5 out-of-distribution.
   - C: Deformable DETR (COCO init) through `automodel`.
   - Epoch budget is matched in wall-clock, not epochs, and stated in the ledger.
2. **Resolution** on the Phase-1 winner(s) plus R50 (c6ecec12 at 800/1000/1200 px is already being re-predicted by
   dense-flock job 52491503 part 0c: `/blue/ewhite/b.weinstein/BOEM/dense_flock/phase0/0c_*.density_strata.csv`; read it first): `min_size` 1000 (native) and ~1600 (upsampled for
   the ViT); optionally a 16 px anchor level (dense-flock anchor audit: 18-35 px birds get ~3 positive anchors vs 20-30 for >35 px,
   but the density drop holds inside every size band, so size is not the whole dense-flock story).
   Dense-flock Phase 1 (52493454): weighted sampler no better than random, dense-only +1.5-2.2 pts at 6-50 birds/tile,
   so sampling changes are NOT carried over. Training on 1000/1200 px (not just inference) is the Phase 2 question. Only arms that beat the gate in Phase 1.
3. **Promotion gate**: beats c6ecec12 at matched precision on BOTH validation and zero-shot beyond
   seed noise (≥2 seeds for the winner), acceptable survey throughput, threshold re-derived with
   `threshold_sweep.py`, end-to-end pipeline run on the debug flight, and one DeepForest ref that carries
   both #1473 and #1334.

## Open choices (defaults in bold)
- DINOv3 weights: **sat493m** (aerial pretraining, used in E007) vs lvd1689m (web images, maybe better on birds/foam).
- DETR flavour: **Deformable DETR** (multi-scale, better on small objects than plain DETR); RT-DETR / RF-DETR later.
- Compute per DINOv3 arm: **2x B200** (E007 setup).

## Reference files for scoring (2026-11-24)
- Candidates: `/blue/ewhite/b.weinstein/BOEM/architecture_comparison/scores/<label>/` (submit_score_detector.sh):
  DINOv3 6fc49164 epoch06 / epoch14 = jobs 52645770 / 52645776; Deformable DETR 5b01a721 after 52511275_1.
- **R50 @1000 retrain (same data/recipe/scale; THE reference)**, e0cb9f95 epoch29, dense-flock job 52653059, in
  `/blue/ewhite/b.weinstein/BOEM/dense_flock/score_at_size/`: `e0cb9f95_e29_in1000.val_predictions.csv`,
  `.val.density_strata.csv`, `.train_predictions.csv`, `.train.density_strata.csv`, `threshold_sweep_e0cb9f95_e29_in1000.json`,
  `.july_summary.csv`. Check its log prints "transform (1000,)/1000".
- c6ecec12 predicted at 1000 (no retrain): validation `/blue/ewhite/b.weinstein/BOEM/dense_flock/phase0/0c_native1000.density_strata.csv`,
  zero-shot `/blue/ewhite/b.weinstein/BOEM/zero_shot_input_size/threshold_sweep_c6ecec12_e27_in1000.json`, train fit / July
  `score_at_size/c6ecec12_e27_in1000.{train.density_strata,july_summary}.csv`.
- c6ecec12 production @800: validation 52344877 predictions next to its checkpoint; zero-shot
  `/blue/ewhite/b.weinstein/BOEM/yolo26_comparison/threshold_sweep_c6ecec12_e27.json`.
