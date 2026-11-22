"""Where does a detection training epoch go? Data loading vs GPU compute.

Builds the model, train CSV and augmentations exactly as USGS_backbone.py does,
then times separately:
  A. the DataLoader alone (decode + augment + collate) at several worker counts;
  B. GPU training steps on batches already resident on the GPU (no I/O), in
     fp32/TF32 (current) and fp16 autocast.
Throughput = images/s. Epoch time ~= n_train / min(A, B), so whichever is lower
is the bottleneck, and B is the per-GPU ceiling more CPUs can buy back.

    uv run python scripts/benchmark_detection_training.py --workers 8 16 24 32
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from deepforest import main
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent))
import kornia.augmentation as K  # noqa: E402,F401  -- same registration side effect as USGS_backbone
from deepforest import augmentations as df_aug  # noqa: E402

CROPS = "/blue/ewhite/b.weinstein/BOEM/training/crops"
TRAIN = f"{CROPS}/train_max_empty_0_10.csv"
AUGS = [{"Rotate": {"degrees": 15, "p": 0.5}}, {"HorizontalFlip": {"p": 0.5}},
        {"VerticalFlip": {"p": 0.3}}, {"RandomAffine": {"degrees": 0, "scale": (0.9, 1.1), "p": 0.3}}]


def build(batch_size: int, workers: int) -> main.deepforest:
    df_aug._SUPPORTED_TRANSFORMS["RandomAffine"] = (K.RandomAffine, {"degrees": 0, "scale": (0.9, 1.1), "p": 0.2})
    m = main.deepforest(config_args={"score_thresh": 0.4})
    m.load_model("weecology/deepforest-bird")
    m.label_dict, m.numeric_to_label_dict = {"Object": 0}, {0: "Object"}
    m.config["train"]["csv_file"], m.config["train"]["root_dir"] = TRAIN, CROPS
    m.config["train"]["validate_coordinates"] = False
    m.config["batch_size"], m.config["workers"] = batch_size, workers
    m.config = OmegaConf.merge(m.config, {"train": {"augmentations": AUGS}})
    return m


def time_loader(m: main.deepforest, n_batches: int) -> float:
    it = iter(m.train_dataloader())
    next(it)  # worker spin-up excluded
    t0 = time.perf_counter()
    n = sum(len(next(it)[0]) for _ in range(n_batches))
    return n / (time.perf_counter() - t0)


def time_gpu(m: main.deepforest, n_steps: int, amp: bool) -> float:
    model = m.model.cuda().train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    it = iter(m.train_dataloader())
    batches = []
    for _ in range(4):
        images, targets, _ = next(it)
        batches.append(([i.cuda() for i in images],
                        [{k: v.cuda() for k, v in t.items()} for t in targets]))

    def step(images, targets):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = sum(model(images, targets).values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for b in batches:  # warm-up / cudnn autotune
        step(*b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_steps):
        step(*batches[i % len(batches)])
    torch.cuda.synchronize()
    return n_steps * len(batches[0][0]) / (time.perf_counter() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, nargs="+", default=[8, 16, 24, 32])
    ap.add_argument("--loader-batches", type=int, default=40)
    ap.add_argument("--gpu-steps", type=int, default=30)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    n_train = 240_635

    print(f"GPU: {torch.cuda.get_device_name(0)} | batch {args.batch_size} | train tiles {n_train:,}", flush=True)
    gpu = {}
    for amp in (False, True):
        gpu[amp] = time_gpu(build(args.batch_size, 4), args.gpu_steps, amp)
        print(f"[B] GPU-only train step, {'fp16 autocast' if amp else 'fp32/TF32 (current)'}: "
              f"{gpu[amp]:.1f} img/s -> {n_train / gpu[amp] / 60:.1f} min/epoch if never starved", flush=True)
    for w in args.workers:
        r = time_loader(build(args.batch_size, w), args.loader_batches)
        print(f"[A] DataLoader only, {w:2d} workers: {r:.1f} img/s -> {n_train / r / 60:.1f} min/epoch", flush=True)
