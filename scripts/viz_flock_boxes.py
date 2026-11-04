"""Overlay ground truth and detections on dense-flock frames.

Renders, per frame:
  1. the full frame downscaled, with GT and predictions
  2. a native-resolution zoom on the densest GT cluster

Predictions are run once at a low floor so the score band is visible: boxes at or
above --production-score are drawn as "kept", the rest as "discarded by the floor".
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

GT_COLOR = (0, 220, 90)        # green
KEPT_COLOR = (255, 45, 45)     # red   - survives the production floor
DROPPED_COLOR = (255, 170, 0)  # amber - emitted but cut by the floor


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_boxes(draw, boxes, color, width, scale=1.0, off=(0, 0)):
    ox, oy = off
    for x1, y1, x2, y2 in boxes:
        draw.rectangle(
            [(x1 - ox) * scale, (y1 - oy) * scale, (x2 - ox) * scale, (y2 - oy) * scale],
            outline=color, width=width,
        )


def legend(img, lines, size=None):
    size = size or max(14, img.width // 90)
    f = _font(size)
    d = ImageDraw.Draw(img, "RGBA")
    pad = size // 2
    h = (size + pad) * len(lines) + pad
    w = max(d.textlength(t, font=f) for _, t in lines) + size * 3
    d.rectangle([0, 0, w, h], fill=(0, 0, 0, 170))
    y = pad
    for color, text in lines:
        d.rectangle([pad, y + size // 4, pad + size, y + size], fill=color)
        d.text((pad * 2 + size, y), text, fill=(255, 255, 255), font=f)
        y += size + pad
    return img


def densest_window(boxes, win):
    """Top-left corner of the `win`x`win` window containing the most boxes."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    best, bxy = -1, (0, 0)
    for i in range(len(boxes)):
        x0, y0 = cx[i] - win / 2, cy[i] - win / 2
        n = int(((cx >= x0) & (cx < x0 + win) & (cy >= y0) & (cy < y0 + win)).sum())
        if n > best:
            best, bxy = n, (x0, y0)
    return int(max(0, bxy[0])), int(max(0, bxy[1])), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--patch", type=int, default=1000)
    ap.add_argument("--predict-score", type=float, default=0.05)
    ap.add_argument("--production-score", type=float, default=0.70)
    ap.add_argument("--nms", type=float, default=0.05)
    ap.add_argument("--full-width", type=int, default=2400)
    ap.add_argument("--zoom", type=int, default=1400)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt_all = pd.read_csv(args.gt_csv)

    from deepforest import main as df_main
    m = df_main.deepforest.load_from_checkpoint(args.checkpoint, map_location="cpu")
    if torch.cuda.is_available():
        m.to("cuda")
    m.eval()
    m.model.score_thresh = args.predict_score
    m.config["score_thresh"] = args.predict_score
    m.config["nms_thresh"] = args.nms

    for name in args.images:
        path = os.path.join(args.image_dir, name)
        if not os.path.exists(path):
            print(f"MISSING {path}", flush=True)
            continue
        gt = gt_all[gt_all.image_path == name][["xmin", "ymin", "xmax", "ymax"]].values.astype(float)

        preds = m.predict_tile(path=path, patch_size=args.patch,
                               patch_overlap=0.0, iou_threshold=args.nms)
        if preds is None or len(preds) == 0:
            pb = np.zeros((0, 4)); ps = np.zeros(0)
        else:
            pb = preds[["xmin", "ymin", "xmax", "ymax"]].values.astype(float)
            ps = preds["score"].values.astype(float)
        kept, dropped = pb[ps >= args.production_score], pb[ps < args.production_score]
        print(f"{name}: gt={len(gt)} pred@{args.predict_score}={len(pb)} "
              f"kept@{args.production_score}={len(kept)}", flush=True)

        im = Image.open(path).convert("RGB")
        W, H = im.size

        # ---- full frame ----
        scale = args.full_width / W
        full = im.resize((args.full_width, int(H * scale)), Image.LANCZOS)
        d = ImageDraw.Draw(full)
        draw_boxes(d, gt, GT_COLOR, 2, scale)
        draw_boxes(d, dropped, DROPPED_COLOR, 2, scale)
        draw_boxes(d, kept, KEPT_COLOR, 3, scale)
        legend(full, [
            (GT_COLOR, f"ground truth  ({len(gt)})"),
            (DROPPED_COLOR, f"detected, cut by the {args.production_score} floor  ({len(dropped)})"),
            (KEPT_COLOR, f"kept at {args.production_score}  ({len(kept)})"),
        ])
        fp = os.path.join(args.out_dir, f"{os.path.splitext(name)[0]}_full.png")
        full.save(fp); print("  wrote", fp, flush=True)

        # ---- native-resolution zoom on the densest cluster ----
        if len(gt):
            x0, y0, n = densest_window(gt, args.zoom)
            x0, y0 = min(x0, max(0, W - args.zoom)), min(y0, max(0, H - args.zoom))
            crop = im.crop((x0, y0, min(W, x0 + args.zoom), min(H, y0 + args.zoom)))
            d = ImageDraw.Draw(crop)
            draw_boxes(d, gt, GT_COLOR, 2, 1.0, (x0, y0))
            draw_boxes(d, dropped, DROPPED_COLOR, 2, 1.0, (x0, y0))
            draw_boxes(d, kept, KEPT_COLOR, 3, 1.0, (x0, y0))
            legend(crop, [
                (GT_COLOR, f"ground truth ({n} in this window)"),
                (DROPPED_COLOR, f"cut by the {args.production_score} floor"),
                (KEPT_COLOR, f"kept at {args.production_score}"),
            ], size=20)
            zp = os.path.join(args.out_dir, f"{os.path.splitext(name)[0]}_zoom.png")
            crop.save(zp); print("  wrote", zp, flush=True)


if __name__ == "__main__":
    main()
