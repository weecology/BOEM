---
library_name: deepforest
tags:
- object-detection
- aerial-imagery
- ecology
- seabirds
- marine-mammals
- deepforest
- retinanet
---

# DeepForest marine biodiversity detector

A single-class ("Object") RetinaNet detector for animals in high-resolution aerial survey imagery of coastal and offshore waters: seabirds, waterfowl, marine mammals, sea turtles, and large fish. It finds animals; it does not identify species. It is the detection stage of the [BOEM aerial survey pipeline](https://github.com/weecology/BOEM), where a separate classifier labels each box.

## Usage

```python
from deepforest import main

model = main.deepforest()
model.load_model("weecology/deepforest-marine-biodiversity")
model.config.score_thresh = 0.70  # see "Choosing a score threshold"
boxes = model.predict_tile("frame.jpg", patch_size=1000, patch_overlap=0)
```

## Choosing a score threshold

**Use 0.70.** That is what the BOEM pipeline runs at. The `score_thresh` of 0.4 in `config.json` is a training-time default, not a recommendation; on open water it flags far more empty frames.

**Do not reuse a threshold tuned on an earlier version of this model.** Score distributions differ between checkpoints: this version's scores reach 0.993, the June 2026 version's top out at 0.974. Re-derive the threshold against your own labelled frames whenever the weights change.

## Training

| | |
|---|---|
| Checkpoint | `1968e962113f4c57bc6bf5f07fae1b25` epoch 29 |
| Trained | 2026-09-03 |
| Initialised from | `weecology/deepforest-tree` |
| Data | 196,272 patches of 400 px (404,517 boxes, 19,629 empty patches) from 76 flights. USGS and BOEM aerial surveys, including the July 2026 survey review pass |
| Training | 30 epochs, SGD, lr 0.001, batch 64, half of each batch positive patches; rotation, flip and scale augmentation |

## Evaluation

Measured on two survey flights withheld from training (JPG_20260201_134000 and JPG_20260202_141900): 648 fully reviewed frames, 1,229 annotated animals. A predicted box counts as a match at IoU ≥ 0.4. These are zero-shot results on unseen flights, not validation-split scores.

| Score threshold 0.70 | This version (Sept 2026) | Previous version (June 2026) |
|---|---|---|
| Animals detected (box recall) | **57.9%** | 52.0% |
| Frames with animals in which at least one was detected | **94.2%** | 87.2% |
| Box precision | 86.6% | **89.0%** |
| Highest box recall at any threshold | **71.4%** | 70.2% |
| Empty-water frames flagged, per 1,000 | 1.0 | 1.0 |

At equal precision (about 85%, threshold 0.60 for both), this version detects 61.0% of animals against 56.9% for the previous one.

The previous version probably trained on these two flights; they were only held out from July 2026. That makes the comparison favour the older model, so the gain shown here is likely an underestimate.

## Limitations

- About 29% of animals are never detected at any threshold. Small birds in dense flocks are missed the most: at 0.70 the model finds fewer than six in ten animals overall.
- Breaking waves and foam cause false positives, and so does beach sand. The BOEM pipeline removes land frames with a separate filter.
- Trained on downward-looking RGB frames at survey altitude. Results on other sensors, altitudes, or regions are untested.

## Versions

| Date | Checkpoint | Notes |
|---|---|---|
| 2026-06-18 | `19022856f861432485a1d3911d85e967` | First release. Trained 2026-05-28. |
| 2026-11 | `1968e962113f4c57bc6bf5f07fae1b25` epoch 29 | Retrained on the corpus updated with the July 2026 survey review. |

To keep using the previous version, pin `revision="eb360dc06c2978e390df7e51046039cf36007ce6"` in `load_model`.
