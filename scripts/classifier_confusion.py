"""Confusion matrix for a metadata-free CropModel checkpoint on its own val split.

Originally written for 56e8585 to ask whether Morus bassanus was absorbed into
Somateria mollissima (it was not -- see JOB_LEDGER.md). Now parameterised so a new
checkpoint can be scored the same way and compared.

Second question, added for the NEAQ-free retrain: does the whale/dolphin confusion
survive? On 56e8585, 46 of the 82 Megaptera novaeangliae predictions were NEAQ
Delphinus delphis crops. NEAQ is boat/variable-distance imagery mixed into
fixed-altitude aerial surveys, so the same species spans a different apparent-size
range; the hypothesis is that the classifier learned crop scale as a species cue and
so read small dolphins as large whales. `scripts/USGS_classification.py` now excludes
NEAQ_* crop CSVs, so a post-exclusion checkpoint tests that directly.

Usage:  python scripts/classifier_confusion.py <comet_id>
Run under SLURM with one GPU; writes matrix + per-crop predictions next to the imagery.
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from deepforest.model import CropModel
from PIL import Image
from pytorch_lightning import Trainer
from torchvision.datasets import ImageFolder

ROOT = "/blue/ewhite/b.weinstein/BOEM/training/classification"
DEFAULT_ID = "a3dc30a085f5442393736ecd96b564c5"

# Cetaceans plus the other large-bodied marine classes they get confused with.
CETACEANS = [
    "Delphinus delphis",
    "Tursiops truncatus",
    "Stenella frontalis",
    "Megaptera novaeangliae",
    "Balaenoptera acutorostrata",
    "Balaenoptera physalus",
    "Eubalaena glacialis",
]


class ImagesOnly(torch.utils.data.Dataset):
    """ImageFolder yields (image, label); predict_step reads a 2-tuple as (images, metadata)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0]


comet_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ID
ckpt = f"{ROOT}/checkpoints/buffer_30/{comet_id}.ckpt"
val = f"{ROOT}/crops/val/buffer_30/{comet_id}"
out = f"/blue/ewhite/b.weinstein/BOEM/classifier_confusion_{comet_id[:8]}"
print(f"checkpoint: {ckpt}\nval split : {val}\noutput    : {out}_*.csv\n")

m = CropModel.load_from_checkpoint(ckpt)
val_ds = ImageFolder(root=val, transform=m.get_transform(augmentations=None))

# The checkpoint's own label mapping must agree with the folder ordering, or every
# label below is silently permuted.
assert m.label_dict == val_ds.class_to_idx, (
    f"label_dict mismatch: model has {len(m.label_dict)} classes, "
    f"folders have {len(val_ds.class_to_idx)}"
)

def logit_predict_step(self, batch, batch_idx):
    """CropModel.predict_step softmaxes before returning, and we need what it consumed.

    In float32 the softmax saturates: 2,138 of a3dc30a0's 3,695 val crops come back at
    exactly 1.0 with every competitor underflowed to 0.0, so log() cannot recover the
    logits after the fact. Temperature scaling has to divide the *pre-softmax* values,
    which means capturing them here. Mirrors the parent's batch unpacking exactly.
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        images, _labels, metadata = batch
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        images, metadata = batch
    else:
        images, metadata = batch, None
    return self.forward(images, metadata=metadata)


m.predict_step = types.MethodType(logit_predict_step, m)

trainer = Trainer(devices=1, accelerator="gpu", logger=False, enable_checkpointing=False)
# predict_dataloader sets shuffle=False, so row i of logits is val_ds.samples[i].
results = trainer.predict(m, m.predict_dataloader(ImagesOnly(val_ds)))
logits = torch.cat([r.float().cpu() for r in results]).numpy()
assert logits.shape == (len(val_ds), len(val_ds.classes)), logits.shape

probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
label = probs.argmax(axis=1)
score = probs.max(axis=1)

paths = [p for p, _ in val_ds.samples]
df = pd.DataFrame({
    "path": paths,
    "true": [m.numeric_to_label_dict[y] for _, y in val_ds.samples],
    "pred": [m.numeric_to_label_dict[i] for i in label],
    "score": np.asarray(score),
})
# Crop pixel size on disk = box + 2*expand, so it tracks apparent object size. This is
# the quantity the scale-shortcut hypothesis is about.
sizes = [Image.open(p).size for p in paths]
df["crop_w"] = [w for w, _ in sizes]
df["crop_h"] = [h for _, h in sizes]
df["crop_px"] = np.sqrt(df.crop_w * df.crop_h)
df["base"] = [os.path.basename(p) for p in paths]
df["neaq"] = df.base.str.startswith("NEAQ")
df["ok"] = df.true == df.pred
df.to_csv(f"{out}_predictions.csv", index=False)
pd.crosstab(df.true, df.pred).to_csv(f"{out}_matrix.csv")
# Raw logits, row-aligned to _predictions.csv and column-aligned to val_ds.classes.
# Everything calibration-related (temperature, margin, entropy, any reliability plot under
# any transform) is answerable from this file without touching a GPU again -- which is the
# only reason it is worth the 1 MB.
np.save(f"{out}_logits.npy", logits.astype(np.float32))
with open(f"{out}_classes.txt", "w") as fh:
    fh.write("\n".join(val_ds.classes))

print(f"val crops: {len(df)}  classes: {len(val_ds.classes)}  NEAQ crops: {int(df.neaq.sum())}")
print(f"overall accuracy: {df.ok.mean():.4f}")
if df.neaq.any():
    print(df.groupby("neaq").agg(n=("ok", "size"), acc=("ok", "mean")).to_string())

per = df.groupby("true").agg(n=("ok", "size"), recall=("ok", "mean"), mean_score=("score", "mean"))
wrong = df[~df.ok]
top = wrong.groupby("true").pred.agg(lambda s: f"{s.value_counts().index[0]} x{s.value_counts().iloc[0]}")
per["top_confusion"] = top
print("\n=== per-class recall (worst first) ===")
print(per.sort_values("recall").round(3).fillna("-").to_string())

prec = df.groupby("pred").agg(n_predicted=("ok", "size"), precision=("ok", "mean"),
                              mean_score=("score", "mean"))
src = wrong.groupby("pred").true.agg(lambda s: f"{s.value_counts().index[0]} x{s.value_counts().iloc[0]}")
prec["top_source"] = src
print("\n=== precision view: where do predictions come from ===")
print(prec.sort_values("n_predicted", ascending=False).round(3).fillna("-").to_string())

print("\n=== cetaceans: recall and leakage into Megaptera ===")
sub = df[df.true.isin(CETACEANS)]
if len(sub):
    t = sub.groupby(["true", "neaq"]).agg(
        n=("ok", "size"), recall=("ok", "mean"),
        to_Megaptera=("pred", lambda s: (s == "Megaptera novaeangliae").sum()),
        median_crop_px=("crop_px", "median"))
    print(t.round(3).to_string())
    print("\n--- full cetacean-row confusion ---")
    ct = pd.crosstab(sub.true, sub.pred)
    print(ct.loc[:, ct.sum() > 0].to_string())

print("\n=== everything predicted as a large whale, and what it really was ===")
for whale in ["Megaptera novaeangliae", "Balaenoptera acutorostrata",
              "Balaenoptera physalus", "Eubalaena glacialis"]:
    w = df[df.pred == whale]
    if not len(w):
        print(f"  {whale}: 0 predictions")
        continue
    print(f"  {whale}: {len(w)} predicted, {w.ok.sum()} correct, "
          f"median crop {w.crop_px.median():.0f}px")
    print(w.groupby("true").agg(n=("ok", "size"), median_crop_px=("crop_px", "median"))
          .sort_values("n", ascending=False).round(1).to_string())

print("\n=== crop size by true class, marine mammals (is size a species cue?) ===")
mm = df[df.true.isin(CETACEANS + ["Halichoerus grypus", "Phoca vitulina", "Mola mola"])]
if len(mm):
    print(mm.groupby("true").crop_px.describe()[["count", "25%", "50%", "75%"]].round(1).to_string())

print("\n=== calibration: is cropmodel_score a probability? ===")
# Two different questions, and this checkpoint answers them oppositely. LEVEL: does 0.9
# mean 90% right? RESOLUTION: does a higher score mean more likely right? Only the second
# has held up so far, which is why human_review.review_high is set from a percentile of
# this distribution rather than from a target accuracy.
bins = [0, .3, .5, .6, .7, .8, .9, .95, .99, 1.001]
df["conf_bin"] = pd.cut(df.score, bins, right=False)
rel = df.groupby("conf_bin", observed=True).agg(
    n=("ok", "size"), acc=("ok", "mean"), mean_conf=("score", "mean"))
rel["share"] = rel.n / len(df)
rel["gap"] = rel.mean_conf - rel.acc  # >0 = overconfident
print(rel[["n", "share", "acc", "mean_conf", "gap"]].round(3).to_string())
ece = float((rel.n / len(df) * rel.gap.abs()).sum())
print(f"\n  mean confidence {df.score.mean():.3f} vs accuracy {df.ok.mean():.3f}  ->  ECE {ece:.3f}")
# Rank-only signal: AUROC of the score for separating correct from incorrect. 0.5 would
# mean the score is worthless for triage no matter how it is thresholded.
r = df.score.rank().values
c = df.ok.values.astype(bool)
n_pos, n_neg = int(c.sum()), int((~c).sum())
if n_pos and n_neg:
    auc = (r[c].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print(f"  AUROC(score -> correct) = {auc:.3f}   (0.5 = no triage value at any threshold)")

print("\n=== choosing human_review.review_high ===")
# A review band [review_low, review_high] is worth its cost only if the errors it catches
# are concentrated relative to the base rate. `lift` is that concentration; `%all errs` is
# the recall of the triage. Pick the cut where recall becomes usable, then sanity-check the
# resulting queue size against the real survey pool -- val crops are all real animals and
# contain no foam, so the val band size is an underestimate of the survey band size.
tot_err = int((~df.ok).sum())
base = tot_err / len(df)
print(f"  base error rate: {tot_err}/{len(df)} = {base:.3f}")
print(f"  {'review_high':>11} {'n reviewed':>11} {'%pool':>7} {'errs':>6} {'%all errs':>10} "
      f"{'err rate':>9} {'lift':>6} | {'auto-annot acc':>14}")
for hi in [0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]:
    k = df[(df.score >= 0.3) & (df.score <= hi)]
    a = df[df.score > hi]
    if not len(k) or not len(a):
        continue
    e = int((~k.ok).sum())
    print(f"  {hi:>11.3f} {len(k):>11d} {len(k)/len(df)*100:>6.1f}% {e:>6d} "
          f"{e/tot_err*100:>9.1f}% {e/len(k):>9.3f} {(e/len(k))/base:>5.2f}x | {a.ok.mean():>14.3f}")
print("  (percentiles of this checkpoint's score: "
      + ", ".join(f"p{int(q*100)}={df.score.quantile(q):.4f}" for q in [0.1, 0.2, 0.3, 0.5]) + ")")


# ---------------------------------------------------------------------------
# Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural Networks")
# ---------------------------------------------------------------------------
# One scalar T, fit post-hoc on this val split by minimising NLL with the network frozen.
# softmax(logits / T). T > 1 softens, T < 1 sharpens.
#
# WHAT IT FIXES: the LEVEL. Cross-entropy has no fixed point -- once accuracy plateaus the
# only way left to reduce training NLL is to inflate logit magnitude, so the net keeps
# getting more confident on what it gets right AND on what it gets wrong. That is the whole
# reason 81% of these crops score >= 0.99 against 76% accuracy.
#
# WHAT IT DOES NOT FIX: the RESOLUTION. Dividing by T is monotone in the logits and leaves
# argmax untouched, so cropmodel_label is unchanged and the ORDER of crops by confidence is
# very nearly unchanged (it can only reshuffle crops whose runner-up structure differs).
# The number of errors a fixed review budget catches is a property of that order. So this
# changes which number you put in human_review.review_high, NOT which crops come back.
# The AUROC printed below is the check: if it barely moves, neither does the queue.
#
# The payoff is portability -- a calibrated review_high means the same thing on the next
# checkpoint, instead of needing a fresh sweep every retrain.
y_true = torch.tensor([y for _, y in val_ds.samples], dtype=torch.long)
lg = torch.from_numpy(logits).double()


def fit_temperature(lg, y_true):
    """Minimise NLL over a single scalar. Optimise log T so T stays positive."""
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), y_true)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def calib_stats(p, y_true, n_bins=15):
    """Max-prob accuracy, NLL and equal-width ECE for a probability matrix."""
    conf, pred = p.max(dim=1)
    ok = (pred == y_true).double()
    nll = F.nll_loss(p.clamp_min(1e-12).log(), y_true).item()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if sel.any():
            ece += sel.double().mean().item() * abs(ok[sel].mean().item() - conf[sel].mean().item())
    return ok.mean().item(), nll, ece


def auroc(scores, ok):
    """AUROC of `scores` for predicting a correct label. Rank-based, so any monotone
    recalibration leaves it fixed -- that invariance is the point of printing it."""
    r = pd.Series(scores).rank().values
    n_pos, n_neg = int(ok.sum()), int((~ok).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return (r[ok].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


T = fit_temperature(lg, y_true)
p_raw = torch.softmax(lg, dim=1)
p_cal = torch.softmax(lg / T, dim=1)

print("\n=== temperature scaling ===")
print(f"  fitted T = {T:.4f}   ({'softening -- overconfident' if T > 1 else 'sharpening -- underconfident'})")
print(f"  {'':<12} {'accuracy':>9} {'NLL':>8} {'ECE':>8}")
for name, p in [("uncalibrated", p_raw), ("T-scaled", p_cal)]:
    acc, nll, ece = calib_stats(p, y_true)
    print(f"  {name:<12} {acc:>9.4f} {nll:>8.4f} {ece:>8.4f}")

ok_np = df.ok.values.astype(bool)
raw_conf = p_raw.max(dim=1).values.numpy()
cal_conf = p_cal.max(dim=1).values.numpy()
print(f"\n  AUROC(confidence -> correct):  uncalibrated {auroc(raw_conf, ok_np):.4f}"
      f"   T-scaled {auroc(cal_conf, ok_np):.4f}")
print("  ^ if these match, temperature bought interpretability, not a better review queue.")

print("\n  confidence distribution, uncalibrated -> T-scaled:")
for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"    p{int(q * 100):<3} {np.quantile(raw_conf, q):>8.4f} -> {np.quantile(cal_conf, q):>8.4f}")
print(f"    share >= 0.99: {(raw_conf >= 0.99).mean():.3f} -> {(cal_conf >= 0.99).mean():.3f}")

print("\n=== review_high on the T-SCALED scale ===")
print("  Same rows as the uncalibrated sweep above, re-indexed onto calibrated confidence.")
tot_err = int((~ok_np).sum())
base = tot_err / len(df)
cal = pd.DataFrame({"conf": cal_conf, "ok": ok_np})
print(f"  {'review_high':>11} {'n reviewed':>11} {'%pool':>7} {'errs':>6} {'%all errs':>10} "
      f"{'err rate':>9} {'lift':>6} | {'auto-annot acc':>14}")
for hi in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    k = cal[cal.conf <= hi]
    a = cal[cal.conf > hi]
    if not len(k) or not len(a):
        continue
    e = int((~k.ok).sum())
    print(f"  {hi:>11.2f} {len(k):>11d} {len(k) / len(df) * 100:>6.1f}% {e:>6d} "
          f"{e / tot_err * 100:>9.1f}% {e / len(k):>9.3f} {(e / len(k)) / base:>5.2f}x | {a.ok.mean():>14.3f}")

# Alternative uncertainty scores. These are NOT monotone functions of max-prob, so unlike
# temperature they can genuinely reorder the queue -- the only thing on this page that can.
print("\n=== can a different uncertainty score beat max-prob? (AUROC -> correct) ===")
srt = np.sort(logits, axis=1)
margin = srt[:, -1] - srt[:, -2]  # top1 - top2 logit
logp = torch.log_softmax(lg, dim=1)
neg_entropy = (logp.exp() * logp).sum(dim=1).numpy()  # higher = more certain
for name, sc in [("max-prob", raw_conf), ("logit margin (top1-top2)", margin),
                 ("negative entropy", neg_entropy)]:
    print(f"  {name:<26} {auroc(sc, ok_np):.4f}")
print("  ^ a clear winner here is worth more than any recalibration of max-prob.")

print(f"\nSuggested config: human_review.review_high on T-scaled scores, T={T:.4f}")
print(f"  Store T with the checkpoint. It is fit to THIS checkpoint's logits and does not transfer.")

print(f"\nwrote {out}_matrix.csv, {out}_predictions.csv, {out}_logits.npy")
