"""Probe whether the trained SpatialTemporalEncoder actually contributes signal.

verify_metadata_fallback.py found that real metadata and zeroed metadata give
byte-identical predictions (mean AND max prob delta 0.0000), which would mean the
metadata branch is dead weight rather than a small contributor. The encoder is
Linear(6,32) -> ReLU -> Dropout, so an all-negative pre-activation would make ReLU
emit zeros for every real coordinate and explain that exactly.
"""

import math

import torch
from deepforest.model import CropModel

CKPT = "/blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints/buffer_30/d8995ca8690046ce9dd775fb42f55cb1.ckpt"

# (lat, lon, day_of_year): neaq/Gulf-ish, plus extremes to see if anything fires.
COORDS = [
    (41.4, -70.9, 82.0),    # neaq, March
    (41.4, -70.9, 200.0),   # neaq, July
    (28.5, -89.0, 230.0),   # BRI Gulf, August
    (60.0, -150.0, 180.0),  # far away
    (0.0, 0.0, 1.0),        # null island
    (-45.0, 170.0, 350.0),  # southern hemisphere
]


def _features(metadata):
    """Re-derive the sinusoidal features the encoder builds internally."""
    lat, lon, doy = metadata[:, 0:1], metadata[:, 1:2], metadata[:, 2:3]
    lat_n, lon_n, doy_n = lat / 90.0, lon / 180.0, (doy - 1) / 365.0
    return torch.cat(
        [
            torch.sin(math.pi * lat_n), torch.cos(math.pi * lat_n),
            torch.sin(math.pi * lon_n), torch.cos(math.pi * lon_n),
            torch.sin(2 * math.pi * doy_n), torch.cos(2 * math.pi * doy_n),
        ],
        dim=1,
    )


def main():
    m = CropModel.load_from_checkpoint(CKPT, map_location="cpu")
    m.eval()
    enc = m.metadata_encoder
    lin = enc.mlp[0]

    print("=== encoder Linear(6,32) weights ===")
    print("weight mean |w| :", f"{lin.weight.abs().mean().item():.6f}")
    print("weight max |w|  :", f"{lin.weight.abs().max().item():.6f}")
    print("bias mean/max/min:", f"{lin.bias.mean().item():.6f}",
          f"{lin.bias.max().item():.6f}", f"{lin.bias.min().item():.6f}")

    print("\n=== encoder output per coordinate (eval mode: dropout off) ===")
    with torch.no_grad():
        for lat, lon, doy in COORDS:
            meta = torch.tensor([[lat, lon, doy]])
            pre = lin(_features(meta))
            out = enc(meta)
            print(
                f"lat={lat:7.1f} lon={lon:8.1f} doy={doy:5.1f} | "
                f"pre-ReLU max {pre.max().item():+.4f} | "
                f"nonzero dims {int((out > 0).sum())}/32 | "
                f"out sum {out.sum().item():.6f}"
            )

    print("\n=== metadata half of classifier ===")
    with torch.no_grad():
        e = enc(torch.tensor([[41.4, -70.9, 82.0]]))
        print("encoder embedding all zero:", bool((e == 0).all()))
        W = m.classifier.weight.detach()
        meta_dim = m.config["cropmodel"].get("metadata_dim", 32)
        feat_dim = W.shape[1] - meta_dim
        print("image half mean |w| :", f"{W[:, :feat_dim].abs().mean().item():.6f}")
        print("meta  half mean |w| :", f"{W[:, feat_dim:].abs().mean().item():.6f}")


if __name__ == "__main__":
    main()
