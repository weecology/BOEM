"""Benchmark detection (and optionally classification) throughput on a B200.

Sweeps predict.batch_size over a real flight directory, sampling GPU utilization
while each run executes, and reports images/s, patches/s and MB/s so a full-survey
runtime can be extrapolated.

    uv run python scripts/benchmark_inference.py
    uv run python scripts/benchmark_inference.py --n-images 200 --batch-sizes 64,128,256
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from PIL import Image

from src import detection

Image.MAX_IMAGE_PIXELS = None


def tabulate(rows, headers):
    """Minimal fixed-width table; avoids a dependency on `tabulate`."""
    cells = [[str(c) for c in row] for row in ([headers] + [list(r) for r in rows])]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(c.rjust(w) for c, w in zip(cells[0], widths)), line]
    out += ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in cells[1:]]
    return "\n".join(out)


class GpuSampler(threading.Thread):
    """Poll nvidia-smi for SM utilization and memory on the visible GPU."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.util = []
        self.mem_used = []
        # Not `self._stop`: Thread._stop is an internal method and shadowing it breaks join().
        self._stop_event = threading.Event()

    def run(self):
        query = "utilization.gpu,memory.used,memory.total"
        while not self._stop_event.is_set():
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()
            if out:
                util, used, total = (float(x) for x in out[0].split(","))
                self.util.append(util)
                self.mem_used.append(used)
                self.mem_total = total
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)
        return {
            "gpu_util_mean": statistics.mean(self.util) if self.util else float("nan"),
            "gpu_util_p90": (
                sorted(self.util)[int(0.9 * (len(self.util) - 1))] if self.util else float("nan")
            ),
            "mem_used_mean_gb": statistics.mean(self.mem_used) / 1024 if self.mem_used else float("nan"),
            "mem_total_gb": getattr(self, "mem_total", float("nan")) / 1024,
            "samples": len(self.util),
        }


def patches_per_image(path, patch_size, patch_overlap):
    """Patch count deepforest's tiler produces for one image."""
    width, height = Image.open(path).size
    stride = patch_size - patch_overlap
    n_cols = max(1, -(-(width - patch_overlap) // stride))
    n_rows = max(1, -(-(height - patch_overlap) // stride))
    return n_cols * n_rows, width, height


def gpu_ceiling(model, batch_size, patch_size, iters=20):
    """Forward-pass-only throughput with data loading removed: the GPU's ceiling."""
    device = torch.device("cuda")
    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    batch = torch.rand(batch_size, 3, patch_size, patch_size, device=device)
    with torch.no_grad():
        for _ in range(3):
            model.model(batch)
        torch.cuda.synchronize()
        sampler = GpuSampler()
        sampler.start()
        start = time.perf_counter()
        for _ in range(iters):
            model.model(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    stats = sampler.stop()
    del batch
    torch.cuda.empty_cache()
    return {
        "patches_per_s": iters * batch_size / elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        **stats,
    }


def run_case(model, image_paths, batch_size, workers, patch_size, patch_overlap, crop_model=None):
    # Release the previous case's cached blocks, or its fragmentation is charged to this one.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler()
    sampler.start()
    start = time.perf_counter()
    predictions = detection.predict(
        m=model,
        image_paths=image_paths,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        crop_model=crop_model,
        workers=workers,
    )
    elapsed = time.perf_counter() - start
    stats = sampler.stop()
    n_boxes = 0 if predictions is None else len(predictions)
    return {
        "elapsed_s": elapsed,
        "images_per_s": len(image_paths) / elapsed,
        "n_boxes": n_boxes,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        **stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="/blue/ewhite/b.weinstein/BOEM/imagery/JPG_20260712_100400")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--batch-sizes", default="16,32,64,128,256")
    parser.add_argument("--workers", default="5")
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--patch-overlap", type=int, default=0)
    parser.add_argument("--with-classification", action="store_true",
                        help="Also time detection+CropModel at the default batch size.")
    parser.add_argument("--out", default="benchmark_inference.json")
    args = parser.parse_args()

    config = OmegaConf.load(PROJECT_ROOT / "boem_conf" / "boem_config.yaml")
    checkpoint = config.detection_model.checkpoint

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg found in {args.image_dir}")
    total_images_in_dir = len(image_paths)
    mean_mb = statistics.mean(os.path.getsize(p) for p in image_paths[:200]) / 1024**2
    n_patches, width, height = patches_per_image(image_paths[0], args.patch_size, args.patch_overlap)
    bench_paths = image_paths[: args.n_images]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Flight: {args.image_dir}")
    print(f"  {total_images_in_dir} images, {width}x{height}, mean {mean_mb:.1f} MB/image, "
          f"{n_patches} patches/image at patch_size={args.patch_size}")
    print(f"Checkpoint: {checkpoint}\n")

    # No create_trainer() here: predict_tile builds its own, exactly as the pipeline does.
    model = detection.load(checkpoint=checkpoint)

    # Warm up CUDA context and page cache so the first sweep entry is not penalized.
    # Warm up at the smallest batch being swept, not a fixed one: batch_size counts images
    # (35 patches each), so a hardcoded 32 here would OOM a 24 GB card before any case ran.
    warmup_batch = min(int(b) for b in args.batch_sizes.split(","))
    run_case(model, bench_paths[:5], warmup_batch, 0, args.patch_size, args.patch_overlap)

    rows = []
    results = {"meta": {
        "gpu": torch.cuda.get_device_name(0),
        "image_dir": args.image_dir,
        "images_in_dir": total_images_in_dir,
        "mean_mb_per_image": mean_mb,
        "patches_per_image": n_patches,
        "resolution": [width, height],
        "n_images_benchmarked": len(bench_paths),
        "checkpoint": checkpoint,
    }, "sweep": [], "ceiling": [], "classification": None}

    worker_counts = [int(w) for w in str(args.workers).split(",")]
    for batch_size in (int(b) for b in args.batch_sizes.split(",")):
        for workers in worker_counts:
            # dataloader_strategy="batch" batches *images*, so one forward pass sees
            # batch_size * patches_per_image patches. Memory climbs fast; keep going
            # past an OOM so the smaller batches still get reported.
            try:
                stats = run_case(model, bench_paths, batch_size, workers,
                                 args.patch_size, args.patch_overlap)
            except torch.OutOfMemoryError:
                print(f"batch={batch_size} workers={workers}: CUDA OOM "
                      f"({batch_size * n_patches} patches/forward pass)")
                torch.cuda.empty_cache()
                results["sweep"].append({"batch_size": batch_size, "workers": workers, "oom": True})
                continue
            stats["batch_size"] = batch_size
            stats["workers"] = workers
            stats["patches_per_s"] = stats["images_per_s"] * n_patches
            stats["mb_per_s"] = stats["images_per_s"] * mean_mb
            stats["hours_per_tb"] = 1024**2 / stats["mb_per_s"] / 3600
            results["sweep"].append(stats)
            rows.append([
                batch_size, workers, f"{stats['elapsed_s']:.1f}",
                f"{stats['images_per_s']:.2f}", f"{stats['patches_per_s']:.0f}",
                f"{stats['mb_per_s']:.0f}", f"{stats['gpu_util_mean']:.0f}%",
                f"{stats['gpu_util_p90']:.0f}%", f"{stats['peak_mem_gb']:.1f}",
                f"{stats['mem_used_mean_gb']:.1f}/{stats['mem_total_gb']:.0f}",
                f"{stats['hours_per_tb']:.1f}",
            ])
            print(tabulate([rows[-1]], headers=[
                "batch", "workers", "sec", "img/s", "patch/s", "MB/s",
                "util avg", "util p90", "torch GB", "smi GB", "h/TB"]))

    print("\n=== Full sweep (detection only) ===")
    print(tabulate(rows, headers=[
        "batch", "workers", "sec", "img/s", "patch/s", "MB/s",
        "util avg", "util p90", "torch GB", "smi GB", "h/TB"]))

    print("\n=== GPU ceiling: forward pass only, no data loading ===")
    ceiling_rows = []
    for batch_size in (int(b) for b in args.batch_sizes.split(",")):
        c = gpu_ceiling(model, batch_size, args.patch_size)
        c["batch_size"] = batch_size
        c["images_per_s"] = c["patches_per_s"] / n_patches
        results["ceiling"].append(c)
        ceiling_rows.append([
            batch_size, f"{c['patches_per_s']:.0f}", f"{c['images_per_s']:.2f}",
            f"{c['gpu_util_mean']:.0f}%", f"{c['peak_mem_gb']:.1f}",
            f"{c['mem_total_gb']:.0f}",
        ])
    print(tabulate(ceiling_rows, headers=[
        "batch", "patch/s", "img/s equiv", "util avg", "torch GB", "total GB"]))

    if args.with_classification:
        from deepforest.model import CropModel
        crop_model = CropModel.load_from_checkpoint(config.classification_model.checkpoint)
        crop_model.config["cropmodel"]["expand"] = config.classification_model.expand
        default_batch = config.predict.batch_size
        stats = run_case(model, bench_paths, default_batch, worker_counts[0],
                         args.patch_size, args.patch_overlap, crop_model=crop_model)
        stats["patches_per_s"] = stats["images_per_s"] * n_patches
        stats["mb_per_s"] = stats["images_per_s"] * mean_mb
        stats["hours_per_tb"] = 1024**2 / stats["mb_per_s"] / 3600
        results["classification"] = stats
        print("\n=== Detection + classification (CropModel) ===")
        print(tabulate([[
            default_batch, f"{stats['elapsed_s']:.1f}", f"{stats['images_per_s']:.2f}",
            f"{stats['mb_per_s']:.0f}", f"{stats['gpu_util_mean']:.0f}%",
            f"{stats['peak_mem_gb']:.1f}", f"{stats['n_boxes']}",
            f"{stats['hours_per_tb']:.1f}",
        ]], headers=["batch", "sec", "img/s", "MB/s", "util avg", "torch GB", "boxes", "h/TB"]))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
