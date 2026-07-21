"""Re-classify the cached July detection boxes with the April (pre-metadata) classifier.

Why: the current classifier (d8995ca8, metadata/PR#1334) produced 10 gray seals and ZERO
harbor seals across ~20k images, while the April run reported 241 Phoca vitulina. The
detection boxes are already cached, so we can swap only the classifier and compare
old-vs-new labels on *identical* boxes — which both tests the "d8995 broke seals"
hypothesis and yields harbor-seal boxes we can upload for review.

April classifier per git 1590e4e (2026-04-27, force_train: False):
    buffer_30/4c002d6bfb654e10ab9ae99aa0451f81.ckpt   (no use_metadata)
"""
import argparse
import datetime
import glob
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch

from deepforest.model import CropModel
from deepforest import predict as df_predict

PINNIPED = "Phoc|Halichoerus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-after", default="2026-07-15",
                    help="only use prediction caches modified on/after this date")
    ap.add_argument("--limit-images", type=int, default=None)
    args = ap.parse_args()

    cut = datetime.datetime.strptime(args.cache_after, "%Y-%m-%d").timestamp()
    base = "/blue/ewhite/b.weinstein/BOEM"
    caches = glob.glob(f"{base}/screened_images/*/.prediction_cache/pool_predictions.csv") + \
             glob.glob(f"{base}/imagery/*/.prediction_cache/pool_predictions.csv")
    caches = [c for c in caches if os.path.getmtime(c) >= cut]
    print(f"fresh caches: {len(caches)}", flush=True)

    frames = []
    for c in caches:
        try:
            d = pd.read_csv(c)
        except Exception:
            continue
        if d.empty or "image_path" not in d.columns:
            continue
        d["flight"] = c.split("/")[-3]
        # The cache stores image_path as a bare basename; the flight's image dir is the
        # cache's grandparent ({image_dir}/.prediction_cache/pool_predictions.csv).
        image_dir = os.path.dirname(os.path.dirname(c))
        d["full_path"] = d["image_path"].astype(str).apply(
            lambda b: b if os.path.isabs(b) else os.path.join(image_dir, b)
        )
        frames.append(d)
    boxes = pd.concat(frames, ignore_index=True)
    print(f"cached boxes: {len(boxes)} across {boxes['full_path'].nunique()} images", flush=True)
    n_exist = boxes["full_path"].map(os.path.exists).sum()
    print(f"boxes whose image resolves on disk: {n_exist}", flush=True)

    crop_model = CropModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    labels = list((crop_model.label_dict or {}).keys())
    print(f"OLD model classes: {len(labels)}", flush=True)
    print(f"OLD model seal labels: {[l for l in labels if 'Phoc' in l or 'Halichoerus' in l]}", flush=True)

    crop_model.create_trainer(logger=False, enable_progress_bar=False,
                             accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=1)
    trainer = crop_model.trainer

    out_frames = []
    groups = list(boxes.groupby("full_path"))
    if args.limit_images:
        groups = groups[:args.limit_images]
    n_missing = 0
    n_failed = 0
    for i, (path, sub) in enumerate(groups):
        if not os.path.exists(path):
            n_missing += 1
            continue
        sub = sub.reset_index(drop=True)
        new_label = sub["cropmodel_label"].tolist() if "cropmodel_label" in sub.columns else [None] * len(sub)
        new_score = sub["cropmodel_score"].tolist() if "cropmodel_score" in sub.columns else [None] * len(sub)
        # BoundingBoxDataset resolves each box against root_dir=dirname(path), so it needs
        # image_path as the basename (which is exactly how the cache stores it).
        work = sub[["xmin", "ymin", "xmax", "ymax", "score"]].copy()
        work["image_path"] = os.path.basename(path)
        try:
            res = df_predict._predict_crop_model_(
                crop_model=crop_model, trainer=trainer, results=work,
                path=path, is_single_model=True,
            )
        except Exception as e:
            # Fail fast: a systematic error (wrong columns, bad paths) otherwise skips every
            # image and the job "succeeds" with an empty CSV. Only tolerate sporadic failures.
            n_failed += 1
            if not out_frames and n_failed >= 3:
                raise RuntimeError(
                    f"first {n_failed} images all failed to classify; aborting rather than "
                    f"writing an empty result. Last error on {path}: {type(e).__name__}: {e}"
                ) from e
            print(f"  [skip] {os.path.basename(path)}: {type(e).__name__} {e}", flush=True)
            continue
        res["image_path"] = path
        res["flight"] = sub["flight"].iloc[0]
        res["old_label"] = res["cropmodel_label"]
        res["old_score"] = res["cropmodel_score"]
        res["new_label"] = new_label[:len(res)]
        res["new_score"] = new_score[:len(res)]
        out_frames.append(res.drop(columns=["cropmodel_label", "cropmodel_score"]))
        if i % 200 == 0:
            print(f"  {i}/{len(groups)} images", flush=True)

    allres = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    allres.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(allres)} boxes "
          f"({n_missing} images skipped: not on disk)", flush=True)

    if len(allres):
        old_seal = allres[allres["old_label"].astype(str).str.contains(PINNIPED, na=False)]
        new_seal = allres[allres["new_label"].astype(str).str.contains(PINNIPED, na=False)]
        print("\n=== OLD (April 4c002d6b) seal calls ===", flush=True)
        print(old_seal["old_label"].value_counts().to_string() or "(none)", flush=True)
        print(f"  unique images: {old_seal['image_path'].nunique()}", flush=True)
        print("\n=== NEW (d8995) seal calls on the SAME boxes ===", flush=True)
        print(new_seal["new_label"].value_counts().to_string() or "(none)", flush=True)
        print(f"  unique images: {new_seal['image_path'].nunique()}", flush=True)
        print("\n=== what NEW calls the boxes OLD says are harbor seal ===", flush=True)
        hs = allres[allres["old_label"].astype(str) == "Phoca vitulina"]
        print(hs["new_label"].value_counts().head(12).to_string() or "(none)", flush=True)


if __name__ == "__main__":
    main()
