# `box_recall` denominator counts empty frames, creating a structural ceiling

## Problem

In `deepforest.metrics.RecallPrecision` (v2.1.0), `box_recall` is computed as:

```python
# metrics.py:201
f"{self.task}_recall": self.recall.float() / self.num_images.float(),
```

But `self.num_images` increments unconditionally on every image (line 99), while empty ground-truth frames early-return on line 112 *before* contributing to `self.recall`:

```python
# metrics.py:99-112
self.num_images += 1
...
is_empty_frame = n_target == 0 or torch.all(target[self.pred_key] == 0)
if is_empty_frame:
    self.num_empty_frames += 1
    if n_pred == 0:
        self.correct_empty_predictions += 1
    else:
        self.num_images_with_predictions += 1
    return   # never reaches self.recall += recall
```

So empty frames contribute **0 to the numerator but 1 to the denominator**, even when the model correctly predicts no boxes on them. This puts a hard ceiling on `box_recall` equal to `num_nonempty / num_total`.

## Minimal example

Consider a validation set with 100 images: 75 empty, 25 with ground-truth boxes. Suppose the model is perfect — it predicts no boxes on every empty frame and recovers every box on every non-empty frame.

Expected:
- `empty_frame_accuracy = 75 / 75 = 1.00`
- `box_recall = 1.00` (every GT box recovered)

Actual:
- `empty_frame_accuracy = 1.00`
- `box_recall = 25 / 100 = 0.25`

The "missing" 0.75 of recall is purely the empty frames being charged against the denominator.

## Why this matters

Aerial / wildlife / remote-sensing datasets are often dominated by empty frames. Users tuning empty-frame ratios (e.g. via a `max_empty_fraction` flag) will see `box_recall` shift purely from changing class balance, with no actual change in detector quality on non-empty images. The metric also looks alarmingly worse than the offline `evaluate.evaluate_geometry`, which strips empty rows and iterates only over images with ground truth.

## Suggested fix

Either:
1. Change the denominator to exclude empty frames:
   ```python
   denom = self.num_images - self.num_empty_frames
   f"{self.task}_recall": self.recall.float() / denom.float() if denom > 0 else torch.tensor(float("nan"))
   ```

