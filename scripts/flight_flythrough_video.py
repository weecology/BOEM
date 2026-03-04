#!/usr/bin/env python3
"""
Generate a low-resolution flythrough video of a BOEM imagery flight.

Uses cached predictions from flight_dir/.prediction_cache/pool_predictions.csv.
Long stretches without detections (>1 min of video time) are cut and replaced
by a brief black card stating how many frames were skipped, followed by a
10-second runway leading into the next detection.
"""

import argparse
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("/blue/ewhite/b.weinstein/BOEM/flight_videos")
_FNAME_RE = re.compile(r"(C\d+)_L\d+_F(\d+)_(T\d{8}_\d{6}_\d{3})", re.IGNORECASE)


def _parse_filename(name):
    m = _FNAME_RE.match(name)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def _load_image(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read()
            rgb = np.rollaxis(data[:3], 0, 3) if data.shape[0] >= 3 else np.stack([data[0]] * 3, axis=-1)
            if rgb.dtype != np.uint8:
                p2, p98 = np.percentile(rgb, (2, 98))
                rgb = np.clip((rgb.astype(float) - p2) / max(p98 - p2, 1e-6) * 255, 0, 255).astype(np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _letterbox(img_w, img_h, out_w, out_h):
    scale = min(out_w / img_w, out_h / img_h)
    nw, nh = int(round(img_w * scale)), int(round(img_h * scale))
    return scale, (out_w - nw) // 2, (out_h - nh) // 2


def _resize_frame(frame, out_w, out_h):
    h, w = frame.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    if nw == out_w and nh == out_h:
        return resized
    out = np.full((out_h, out_w, 3), 28, dtype=np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def _draw_arrow(frame, target_x, target_y):
    """Compact arrow that stops 80px short of the target so it does not obscure it."""
    h, w = frame.shape[:2]
    offset, arrow_len = 80, 180
    dx, dy = target_x - w // 2, target_y - h // 4
    dist = math.hypot(dx, dy) or 1
    ux, uy = dx / dist, dy / dist
    end_x = int(target_x - ux * offset)
    end_y = int(target_y - uy * offset)
    start_x = int(end_x - ux * arrow_len)
    start_y = int(end_y - uy * arrow_len)
    cv2.arrowedLine(
        frame, (start_x, start_y), (end_x, end_y),
        (0, 210, 255), 3, tipLength=0.25, line_type=cv2.LINE_AA,
    )


def _make_skip_card(out_w, out_h, n_skipped):
    card = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    l1 = f"Skipping {n_skipped:,} frames"
    l2 = "(no detections)"
    (tw1, th1), _ = cv2.getTextSize(l1, font, 1.4, 2)
    (tw2, _), _ = cv2.getTextSize(l2, font, 0.9, 1)
    cv2.putText(card, l1, ((out_w - tw1) // 2, out_h // 2 - 10),
                font, 1.4, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(card, l2, ((out_w - tw2) // 2, out_h // 2 + th1 + 10),
                font, 0.9, (130, 130, 130), 1, cv2.LINE_AA)
    return card


def _zoomed_crop(frame, preds, orig_w, orig_h,
                 zoom_factor=1.5, min_crop=300, context_pad=0.8):
    """Crop around detections with generous context, zoom, overlay labels."""
    xs = preds[["xmin", "xmax"]].values.flatten()
    ys = preds[["ymin", "ymax"]].values.flatten()
    det_w, det_h = xs.max() - xs.min(), ys.max() - ys.min()
    pad_w = max(det_w * context_pad, min_crop)
    pad_h = max(det_h * context_pad, min_crop)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max((det_w + 2 * pad_w) / 2, (det_h + 2 * pad_h) / 2)
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(orig_w, int(cx + half))
    y2 = min(orig_h, int(cy + half))
    x1, y1 = max(0, x2 - int(2 * half)), max(0, y2 - int(2 * half))

    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return frame
    zoomed = cv2.resize(crop, None, fx=zoom_factor, fy=zoom_factor,
                        interpolation=cv2.INTER_LINEAR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for _, row in preds.iterrows():
        rxmin = int(round((row["xmin"] - x1) * zoom_factor))
        rymin = int(round((row["ymin"] - y1) * zoom_factor))
        rxmax = int(round((row["xmax"] - x1) * zoom_factor))
        rymax = int(round((row["ymax"] - y1) * zoom_factor))
        cv2.rectangle(zoomed, (rxmin, rymin), (rxmax, rymax), (0, 255, 160), 2)

        det_s = float(row.get("score", 0))
        cls_s = row.get("cropmodel_score", np.nan)
        cls_l = (str(row.get("cropmodel_label", ""))[:24]
                 if pd.notna(row.get("cropmodel_label")) else "")
        txt1 = (f"det:{det_s:.2f} cls:{float(cls_s):.2f}"
                if pd.notna(cls_s) else f"det:{det_s:.2f}")
        txt2 = cls_l

        (_, th1), _ = cv2.getTextSize(txt1, font, 0.55, 1)
        (_, th2), _ = cv2.getTextSize(txt2, font, 0.55, 1)
        ty = rymin - 6
        if ty - th1 - th2 - 4 < 0:
            ty = rymax + th1 + th2 + 10
        if txt2:
            cv2.putText(zoomed, txt1, (rxmin, ty - th2 - 4),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(zoomed, txt2, (rxmin, ty),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(zoomed, txt1, (rxmin, ty),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return zoomed


def _paste_zoom(base, zoomed, margin=20):
    out = base.copy()
    oh, ow = out.shape[:2]
    zh, zw = zoomed.shape[:2]
    if zw > ow - 2 * margin or zh > oh - 2 * margin:
        s = min((ow - 2 * margin) / zw, (oh - 2 * margin) / zh)
        zoomed = cv2.resize(zoomed, None, fx=s, fy=s,
                            interpolation=cv2.INTER_LINEAR)
        zh, zw = zoomed.shape[:2]
    x0, y0 = (ow - zw) // 2, (oh - zh) // 2
    cv2.rectangle(out, (x0 - 2, y0 - 2), (x0 + zw + 2, y0 + zh + 2),
                  (0, 255, 160), 2)
    out[y0:y0 + zh, x0:x0 + zw] = zoomed
    return out


def _plan_frames(has_det, fps, skip_gap_sec=60.0,
                 runway_sec=10.0, trail_sec=2.0):
    """Decide which thinned frames to include and where to insert skip cards."""
    n = len(has_det)
    skip_threshold = int(fps * skip_gap_sec)
    runway_n, trail_n = int(fps * runway_sec), int(fps * trail_sec)
    det_indices = [i for i, d in enumerate(has_det) if d]
    include = [False] * n
    skip_cards = {}

    if not det_indices:
        return [True] * n, {}

    def mark(a, b):
        for j in range(max(0, a), min(n, b)):
            include[j] = True

    gap = det_indices[0]
    if gap <= skip_threshold:
        mark(0, det_indices[0])
    else:
        rs = max(0, det_indices[0] - runway_n)
        mark(rs, det_indices[0])
        if rs > 0:
            skip_cards[rs] = rs

    for k in range(len(det_indices)):
        include[det_indices[k]] = True
        if k + 1 >= len(det_indices):
            break
        gs, ge = det_indices[k] + 1, det_indices[k + 1]
        gap = ge - gs
        if gap <= skip_threshold:
            mark(gs, ge)
        else:
            te = min(gs + trail_n, ge)
            mark(gs, te)
            rs = max(te, ge - runway_n)
            mark(rs, ge)
            skipped = rs - te
            if skipped > 0:
                skip_cards[rs] = skipped

    mark(det_indices[-1] + 1, min(det_indices[-1] + 1 + trail_n, n))
    return include, skip_cards


def generate_flythrough(
    flight_dir, output_path=None, output_dir=None, camera="auto",
    width=1280, height=720, fps=10, thin=5,
    pause_seconds=2.0, zoom_seconds=2.5,
    skip_card_seconds=5.0, min_score=0.3,
):
    flight_dir = Path(flight_dir).resolve()
    if not flight_dir.is_dir():
        raise FileNotFoundError(f"Flight directory not found: {flight_dir}")
    all_images = list(flight_dir.glob("*.jpg")) + list(flight_dir.glob("*.JPG"))
    if not all_images:
        raise FileNotFoundError(f"No images in {flight_dir}")

    parsed = []
    for p in all_images:
        info = _parse_filename(p.name)
        parsed.append((p, info[0], info[1]) if info else (p, "__unknown__", 0))

    pred_path = flight_dir / ".prediction_cache" / "pool_predictions.csv"
    if not pred_path.is_file():
        raise FileNotFoundError(f"No cached predictions at {pred_path}")
    pred_df = pd.read_csv(pred_path)
    pred_df["_bn"] = pred_df["image_path"].astype(str).apply(os.path.basename)
    above = pred_df[pred_df["score"].astype(float) >= min_score]
    total_boxes = len(above)
    det_bns = set(above["_bn"])
    n_images_with_det = len(det_bns)
    print(f"Cache: {total_boxes} detection boxes in {n_images_with_det} unique images (across all cameras).")

    if camera == "auto":
        # Pick the camera that has the most *images* (frames) with at least one detection.
        cam_det = {}
        for p, cam, _ in parsed:
            if p.name in det_bns:
                cam_det[cam] = cam_det.get(cam, 0) + 1
        if cam_det:
            camera = max(cam_det, key=cam_det.get)
            print(f"Camera {camera}: auto-selected (has most images with detections: "
                  f"{cam_det[camera]} of this camera's images have >=1 box).")
        else:
            cams = {}
            for _, cam, _ in parsed:
                cams[cam] = cams.get(cam, 0) + 1
            camera = max(cams, key=cams.get)
            print(f"No detections; defaulting to camera {camera}.")

    cam_imgs = sorted(
        [(p, fnum) for p, cam, fnum in parsed if cam == camera],
        key=lambda x: x[1],
    )
    images = [p for p, _ in cam_imgs]
    print(f"Camera {camera}: {len(images)} images total.")

    # Never drop frames that have detections: thinned list plus every frame that has a detection.
    thinned = images[::thin]
    detection_frames = [p for p in images if p.name in det_bns]
    frame_list = sorted(
        set(thinned) | set(detection_frames),
        key=lambda p: images.index(p),
    )
    has_det = [p.name in det_bns for p in frame_list]
    n_det_frames = sum(has_det)
    print(f"Frames to consider: {len(frame_list)} (thinned + all detection frames; {n_det_frames} have >=1 detection).")

    include, skip_cards = _plan_frames(has_det, fps)
    print(f"Rendering {sum(include)} frames, "
          f"{len(skip_cards)} skip cards "
          f"({sum(skip_cards.values()):,} skipped)")

    if output_path is None:
        out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{flight_dir.name}_flythrough.avi"
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if output_path.suffix.lower() != ".avi":
            output_path = output_path.with_suffix(".avi")

    out_w, out_h = width, height
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {output_path}")

    # Show arrow at the start for pause_seconds, then zoom (no separate pause then arrow).
    n_arrow_frames = max(1, int(round(pause_seconds * fps)))
    n_zoom = max(1, int(round(zoom_seconds * fps)))
    n_skip = max(1, int(round(skip_card_seconds * fps)))

    try:
        written = 0
        for i, img_path in enumerate(frame_list):
            if not include[i]:
                continue
            if i in skip_cards:
                card = _make_skip_card(out_w, out_h, skip_cards[i])
                for _ in range(n_skip):
                    writer.write(card)

            frame = _load_image(str(img_path))
            if frame is None:
                continue
            fh, fw = frame.shape[:2]
            scale, lx, ly = _letterbox(fw, fh, out_w, out_h)
            resized = _resize_frame(frame, out_w, out_h)

            preds = pred_df[
                (pred_df["_bn"] == img_path.name)
                & (pred_df["score"].astype(float) >= min_score)
            ]

            if preds.empty:
                writer.write(resized)
                written += 1
                continue

            # Arrows for all detections; zoom on one only (first).
            arrow_frame = resized.copy()
            for _, row in preds.iterrows():
                tx = int(round((row["xmin"] + row["xmax"]) / 2 * scale)) + lx
                ty = int(round((row["ymin"] + row["ymax"]) / 2 * scale)) + ly
                _draw_arrow(arrow_frame, tx, ty)
            for _ in range(n_arrow_frames):
                writer.write(arrow_frame)

            zoomed = _zoomed_crop(frame, preds.iloc[:1], fw, fh)
            overlay = _paste_zoom(resized, zoomed)
            for _ in range(n_zoom):
                writer.write(overlay)

            written += 1
            if written % 50 == 0:
                print(f"  {written} frames written ...")
    finally:
        writer.release()

    print(f"Done: {written} frames -> {output_path}")
    return str(output_path)


def main():
    p = argparse.ArgumentParser(
        description="BOEM flight flythrough video with cached predictions")
    p.add_argument("flight_dir")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--output-dir", default=None,
                   help=f"Default: {DEFAULT_OUTPUT_DIR}")
    p.add_argument("--camera", default="auto",
                   help="Camera ID (e.g. C1) or auto")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--thin", type=int, default=5)
    p.add_argument("--pause-seconds", type=float, default=2.0,
                   help="Seconds to show arrow on detection (default: 2)")
    p.add_argument("--zoom-seconds", type=float, default=2.5)
    p.add_argument("--skip-card-seconds", type=float, default=5.0)
    p.add_argument("--min-score", type=float, default=0.3)
    args = p.parse_args()
    out = generate_flythrough(
        flight_dir=args.flight_dir, output_path=args.output,
        output_dir=args.output_dir, camera=args.camera,
        width=args.width, height=args.height, fps=args.fps,
        thin=args.thin, pause_seconds=args.pause_seconds,
        zoom_seconds=args.zoom_seconds,
        skip_card_seconds=args.skip_card_seconds,
        min_score=args.min_score,
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
