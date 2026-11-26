"""E1 (out-of-range audit) and E2 (post-hoc GBIF prior sweep) on cached held-out logits.

Inputs from geo_priors_eval_logits.py and geo_priors_gbif.py. CPU only.
prior_c(cell) -> adjusted logit = logit_c + lam * log(max(prior_c, eps)).
Settings are chosen on one half of the flights and scored on the other (and vice
versa), so reported numbers never use the flights the setting was tuned on.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/blue/ewhite/b.weinstein/BOEM/geo_priors")
boxes = pd.read_csv(D / "eval_boxes_0139af8c.csv", parse_dates=["date"])
logits = np.load(D / "eval_logits_0139af8c.npy")
classes = (D / "eval_classes_0139af8c.txt").read_text().split("\n")
gbif = pd.read_csv(D / "gbif_counts.csv")
cidx = {c: i for i, c in enumerate(classes)}
y = boxes.label.map(cidx).to_numpy()
subs = sorted(gbif.subregion.unique())
season = lambda m: (m % 12) // 3  # Dec-Feb=0, Mar-May=1, Jun-Aug=2, Sep-Nov=3

# counts[c, sub, month-1]
counts = np.zeros((len(classes), len(subs), 12))
for r in gbif[gbif.label != "__effort__"].itertuples():
    counts[cidx[r.label], subs.index(r.subregion), r.month - 1] = r.n
effort = np.zeros((len(subs), 12))
for r in gbif[gbif.label == "__effort__"].itertuples():
    effort[subs.index(r.subregion), r.month - 1] = r.n
missing = [c for c in classes if counts[cidx[c]].sum() == 0]
print("classes with no GBIF records in any sub-region:", missing)


def prior(source: str, grain: str) -> np.ndarray:
    """[class, sub, month] prior in (0, 1]; grain pools months (annual/season/month)."""
    n, e = counts.copy(), effort.copy()
    if grain != "month":
        groups = np.zeros(12, int) if grain == "annual" else np.array([season(m + 1) for m in range(12)])
        for g in set(groups):
            sel = groups == g
            n[..., sel] = n[..., sel].sum(-1, keepdims=True)
            e[..., sel] = e[..., sel].sum(-1, keepdims=True)
    share = n / n.sum((1, 2), keepdims=True).clip(1)  # fraction of class's records in this cell
    if source == "ratio":  # likelihood ratio vs. how much of all GBIF effort is in the cell
        share = share / (e / e.sum()).clip(1e-9)
        return share / share.max((1, 2), keepdims=True).clip(1e-12)
    return share / share.max((1, 2), keepdims=True).clip(1e-12)


sub_i = boxes.subregion.map(subs.index).to_numpy()
mon_i = boxes.date.dt.month.to_numpy() - 1


def adjust(source, grain, lam, eps):
    p = prior(source, grain)[:, sub_i, mon_i].T  # [box, class]
    return logits + lam * np.log(np.maximum(p, eps))


def metrics(z, mask):
    pred = z[mask].argmax(1)
    yy = y[mask]
    per_class = pd.Series(pred == yy).groupby(yy).mean()
    return {"box_acc": (pred == yy).mean(), "macro_acc": per_class.mean(), "n": int(mask.sum())}


# ---- E1: out-of-range audit of the unmodified classifier ----
pred0 = logits.argmax(1)
cell_n = counts.sum(-1)[pred0, sub_i]  # GBIF records of the predicted class in the box's sub-region
boxes["pred"] = np.array(classes)[pred0]
boxes["ok"] = pred0 == y
boxes["oor"] = cell_n < 20
print("\n=== E1: baseline by sub-region (human-labelled boxes) ===")
print(boxes.groupby("subregion").agg(boxes=("ok", "size"), frames=("basename", "nunique"),
      acc=("ok", "mean"), oor_pred=("oor", "mean")).round(3).to_string())
print("\nout-of-range predictions (<20 GBIF records of the predicted class in the sub-region), top absorbers:")
print(boxes[boxes.oor].groupby(["subregion", "pred"]).size().sort_values(ascending=False).head(15).to_string())
print("\nout-of-range predictions where the true label IS in range (the prior's fixable errors):")
true_n = counts.sum(-1)[y, sub_i]
fix = boxes[boxes.oor & (true_n >= 20)]
print(f"{len(fix)} boxes ({len(fix)/len(boxes):.2%}); top true->pred:")
print(fix.groupby(["label", "pred"]).size().sort_values(ascending=False).head(10).to_string())
print("\ntrue label has <20 GBIF records in its own sub-region (prior would hurt):")
print(f"{(true_n < 20).sum()} boxes: {boxes[true_n < 20].label.value_counts().head(8).to_dict()}")

# ---- E2: sweep, tuned on one flight half, scored on the other ----
fold = (pd.util.hash_array(boxes.flight.fillna(boxes.basename).to_numpy()) % 2).astype(int)
grid = list(itertools.product(["share", "ratio"], ["annual", "season", "month"], [0.25, 0.5, 1.0, 2.0], [1e-3, 1e-2, 0.1]))
rows = []
for cfg in grid:
    z = adjust(*cfg)
    for f in (0, 1):
        rows.append({"source": cfg[0], "grain": cfg[1], "lam": cfg[2], "eps": cfg[3], "fold": f,
                     **metrics(z, fold == f)})
res = pd.DataFrame(rows)
res.to_csv(D / "e2_sweep_0139af8c.csv", index=False)

print("\n=== E2: baseline vs best tuned-on-other-half (macro acc across classes) ===")
for f in (0, 1):
    tune = res[res.fold == 1 - f].sort_values("macro_acc", ascending=False).iloc[0]
    test = res[(res.fold == f) & (res.source == tune.source) & (res.grain == tune.grain)
               & (res.lam == tune.lam) & (res.eps == tune.eps)].iloc[0]
    base = metrics(logits, fold == f)
    print(f"fold {f}: tuned on {tune.source}/{tune.grain}/lam={tune.lam}/eps={tune.eps} | "
          f"baseline box={base['box_acc']:.4f} macro={base['macro_acc']:.4f} -> "
          f"prior box={test.box_acc:.4f} macro={test.macro_acc:.4f}")
print("\nfull-sample sweep (mean of folds), top 10 by macro acc; baseline macro =",
      round(metrics(logits, np.ones(len(y), bool))["macro_acc"], 4))
print(res.groupby(["source", "grain", "lam", "eps"])[["box_acc", "macro_acc"]].mean()
      .sort_values("macro_acc", ascending=False).head(10).round(4).to_string())

best = res.groupby(["source", "grain", "lam", "eps"])["macro_acc"].mean().idxmax()
z = adjust(*best)
print("\nbest config", best, "- by region (box acc / macro acc), baseline -> prior")
for name, mask in [("Gulf", (boxes.region == "Gulf").to_numpy()), ("Atlantic", (boxes.region == "Atlantic").to_numpy())]:
    b, a = metrics(logits, mask), metrics(z, mask)
    print(f"{name:9s} n={b['n']:6d} box {b['box_acc']:.4f} -> {a['box_acc']:.4f}   macro {b['macro_acc']:.4f} -> {a['macro_acc']:.4f}")
print("\nper-class change in recall (best config), largest moves:")
pc = pd.DataFrame({"base": pd.Series(pred0 == y).groupby(y).mean(), "prior": pd.Series(z.argmax(1) == y).groupby(y).mean(),
                   "n": pd.Series(y).value_counts()})
pc.index = [classes[i] for i in pc.index]
pc["delta"] = pc.prior - pc.base
print(pc.reindex(pc.delta.abs().sort_values(ascending=False).index).head(12).round(3).to_string())
