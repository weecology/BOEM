"""Diagnose why epoch16 detection under-detects at inference.

Runs one detection checkpoint over a fixed list of images at a very low score
threshold and records every box + score, plus overlay PNGs. Intended to be run
under two different DeepForest venvs (balanced-empty-frames vs friendly-beaver)
so we can tell whether the branch the model is *loaded* under changes its output.
"""
import argparse
import json
import os

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import deepforest
from deepforest import main as df_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-list", required=True, help="txt file, one image basename per line")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True, help="label for this run, e.g. balanced or friendlybeaver")
    ap.add_argument("--min-score", type=float, default=0.01)
    ap.add_argument("--patch-size", type=int, default=1000)
    ap.add_argument("--patch-overlap", type=int, default=0)
    ap.add_argument("--n-overlays", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[{args.tag}] deepforest {deepforest.__version__} from {deepforest.__file__}", flush=True)

    with open(args.image_list) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    paths = [os.path.join(args.image_dir, n) for n in names]
    paths = [p for p in paths if os.path.exists(p)]
    print(f"[{args.tag}] {len(paths)} images exist of {len(names)} listed", flush=True)

    m = df_main.deepforest.load_from_checkpoint(args.checkpoint, map_location="cpu")
    m.config["score_thresh"] = args.min_score
    try:
        m.model.score_thresh = args.min_score
    except Exception:
        pass
    import torch
    if torch.cuda.is_available():
        m.to("cuda")
    m.eval()

    rows = []
    overlays_done = 0
    for p in paths:
        try:
            preds = m.predict_tile(path=p, patch_size=args.patch_size,
                                   patch_overlap=args.patch_overlap)
        except Exception as e:
            print(f"[{args.tag}] predict failed {os.path.basename(p)}: {type(e).__name__} {e}", flush=True)
            preds = None
        n = 0 if preds is None else len(preds)
        if preds is not None and n:
            preds["image"] = os.path.basename(p)
            rows.append(preds[[c for c in ("image", "xmin", "ymin", "xmax", "ymax", "label", "score") if c in preds.columns]])
        print(f"[{args.tag}] {os.path.basename(p)}: {n} boxes", flush=True)

        # overlay first n_overlays images that have boxes
        if preds is not None and n and overlays_done < args.n_overlays:
            try:
                im = Image.open(p).convert("RGB")
                dr = ImageDraw.Draw(im)
                for _, r in preds.iterrows():
                    sc = float(r.get("score", 0))
                    color = (0, 255, 0) if sc >= 0.85 else ((255, 165, 0) if sc >= 0.3 else (255, 0, 0))
                    dr.rectangle([r["xmin"], r["ymin"], r["xmax"], r["ymax"]], outline=color, width=4)
                    dr.text((r["xmin"], max(0, r["ymin"] - 12)), f"{sc:.2f}", fill=color)
                # downscale for artifact
                im.thumbnail((900, 900))
                out = os.path.join(args.out_dir, f"overlay_{args.tag}_{overlays_done:02d}_{os.path.basename(p)}.png")
                im.save(out)
                overlays_done += 1
            except Exception as e:
                print(f"[{args.tag}] overlay failed {os.path.basename(p)}: {e}", flush=True)

    allbox = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["image", "score"])
    allbox.to_csv(os.path.join(args.out_dir, f"boxes_{args.tag}.csv"), index=False)

    summary = {"tag": args.tag, "deepforest_version": deepforest.__version__,
               "n_images": len(paths), "total_boxes": int(len(allbox)),
               "images_with_any_box": int(allbox["image"].nunique()) if len(allbox) else 0}
    for th in (0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85):
        summary[f"boxes_ge_{th}"] = int((allbox["score"] >= th).sum()) if len(allbox) else 0
        summary[f"imgs_ge_{th}"] = int(allbox[allbox["score"] >= th]["image"].nunique()) if len(allbox) else 0
    with open(os.path.join(args.out_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.tag}] SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
