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
- **Dense flocks** (agreed with the dense-flock chat, 2026-11-23): score raw predictions (score_thresh <= 0.3) with
  `scripts/density_strata.py <predictions.csv> [gt.csv]` (from dense-flock-diagnostics commit 55a7050) on the validation
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
1. **Architecture at fixed input (800 px), same split, same augmentations.**
   - B: RetinaNet + DINOv3 ViT-L sat493m, encoder unfrozen (MillionTrees E007 recipe: SGD, cosine, bf16,
     grad clip 1.0). E007 gain over the fine-tuned R50: macro F1 +0.8 within-distribution, +4.5 out-of-distribution.
   - C: Deformable DETR (COCO init) through `automodel`.
   - Epoch budget is matched in wall-clock, not epochs, and stated in the ledger.
2. **Resolution** on the Phase-1 winner(s) plus R50 (c6ecec12 at 800/1000/1200 px is already being re-predicted by
   dense-flock job 52491503 part 0c: `/blue/ewhite/b.weinstein/BOEM/dense_flock/phase0/0c_*.density_strata.csv`; read it first): `min_size` 1000 (native) and ~1600 (upsampled for
   the ViT); optionally a 16 px anchor level (dense-flock anchor audit: 18-35 px birds get ~3 positive anchors vs 20-30 for >35 px,
   but the density drop holds inside every size band, so size is not the whole dense-flock story).
   If dense-flock Phase 1 (52493454) shows the density-weighted sampler helps, apply it to the candidates too. Only arms that beat the gate in Phase 1.
3. **Promotion gate**: beats c6ecec12 at matched precision on BOTH validation and zero-shot beyond
   seed noise (≥2 seeds for the winner), acceptable survey throughput, threshold re-derived with
   `threshold_sweep.py`, end-to-end pipeline run on the debug flight, and one DeepForest ref that carries
   both #1473 and #1334.

## Open choices (defaults in bold)
- DINOv3 weights: **sat493m** (aerial pretraining, used in E007) vs lvd1689m (web images, maybe better on birds/foam).
- DETR flavour: **Deformable DETR** (multi-scale, better on small objects than plain DETR); RT-DETR / RF-DETR later.
- Compute per DINOv3 arm: **2x B200** (E007 setup).
