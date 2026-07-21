"""Verify whether the metadata-trained CropModel can run without metadata.

submit_seals_array.sh warns that checkpoint d8995ca8 "was trained with metadata
(metadata_dim=32) and needs it at inference". BRI and neaq have no flight metadata
CSVs, so this decides whether those datasets can be classified at all.

Answers three questions:
  1. What architecture does load_from_checkpoint build (config yaml vs ckpt hparams)?
  2. Does forward(x, metadata=None) run, or raise?
  3. How far do zero-metadata predictions drift from real-metadata predictions?
"""

import numpy as np
import torch
from deepforest.model import CropModel

CKPT = "/blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints/buffer_30/d8995ca8690046ce9dd775fb42f55cb1.ckpt"


def main():
    print("loading checkpoint...", flush=True)
    m = CropModel.load_from_checkpoint(CKPT, map_location="cpu")
    m.eval()

    cm = m.config["cropmodel"]
    print("\n=== 1. ARCHITECTURE AS LOADED ===")
    print("ckpt hparams use_metadata :", cm.get("use_metadata"))
    print("metadata_dim              :", cm.get("metadata_dim"))
    print("metadata_dropout          :", cm.get("metadata_dropout"))
    print("backbone is None          :", m.backbone is None)
    print("metadata_encoder is None  :", m.metadata_encoder is None)
    print("model is None             :", m.model is None)
    if m.classifier is not None:
        print("classifier                : Linear(%d -> %d)"
              % (m.classifier.in_features, m.classifier.out_features))
    n_classes = len(m.label_dict) if getattr(m, "label_dict", None) else "?"
    print("num classes               :", n_classes)

    # Does the BOEM yaml flag change the architecture? pipeline.py:223 calls
    # load_from_checkpoint with no config override, so it should not.
    print("\n=== 2. FORWARD WITHOUT METADATA (the BRI/neaq case) ===")
    x = torch.randn(4, 3, 224, 224)
    try:
        with torch.no_grad():
            out_none = m.forward(x, metadata=None)
        print("forward(metadata=None)    : OK, logits", tuple(out_none.shape))
    except Exception as e:
        print("forward(metadata=None)    : RAISED", type(e).__name__, e)
        return

    print("\n=== 3. ZERO vs REAL METADATA DRIFT ===")
    # neaq-like coords: 41.4N, -70.9W, day-of-year 82
    meta = torch.tensor([[41.4, -70.9, 82.0]] * 4)
    with torch.no_grad():
        out_real = m.forward(x, metadata=meta)
    p_none = torch.softmax(out_none, 1)
    p_real = torch.softmax(out_real, 1)
    agree = (p_none.argmax(1) == p_real.argmax(1)).float().mean().item()
    print("top-1 agreement (random imgs):", f"{agree:.2%}")
    print("mean abs prob delta          :", f"{(p_none - p_real).abs().mean().item():.4f}")
    print("max abs prob delta           :", f"{(p_none - p_real).abs().max().item():.4f}")

    # How much of the classifier's weight mass sits on the metadata half?
    W = m.classifier.weight.detach()
    feat_dim = W.shape[1] - cm.get("metadata_dim", 32)
    img_norm = W[:, :feat_dim].abs().mean().item()
    meta_norm = W[:, feat_dim:].abs().mean().item()
    print("\nclassifier mean |w| image half :", f"{img_norm:.5f}")
    print("classifier mean |w| meta half  :", f"{meta_norm:.5f}")
    print("meta/image weight ratio        :", f"{meta_norm / img_norm:.3f}")


if __name__ == "__main__":
    main()
