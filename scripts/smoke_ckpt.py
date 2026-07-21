"""Check that both latest-round checkpoints load in the classification-branch DeepForest.

The detection checkpoint (job 36523583, epoch 16/20) was trained on the balanced-empty-frames
branch, while the classification checkpoint (job 36539340) needs claude/friendly-beaver
(PR #1334, metadata embeddings). The seals pipeline loads both in one process, so the
classification-branch DeepForest has to be able to read the detection checkpoint too.
"""
import time
import warnings

warnings.filterwarnings("ignore")

DET = "/blue/ewhite/b.weinstein/BOEM/training/checkpoints/a09c69331af8496380cbf99e3859d656/epoch16-val_cls0.0163.ckpt"
CLS = "/blue/ewhite/b.weinstein/BOEM/training/classification/checkpoints/buffer_30/d8995ca8690046ce9dd775fb42f55cb1.ckpt"

t = time.time()
from deepforest import main as df_main
from deepforest.model import CropModel

print(f"imports OK ({time.time() - t:.1f}s)", flush=True)

print("--- detection ckpt (trained on balanced-empty-frames branch) ---", flush=True)
try:
    m = df_main.deepforest.load_from_checkpoint(DET, map_location="cpu")
    print("OK detection loaded; label_dict =", getattr(m, "label_dict", None), flush=True)
except Exception as e:
    print(f"FAIL detection: {type(e).__name__}: {str(e)[:500]}", flush=True)

print("--- classification ckpt (metadata / PR #1334) ---", flush=True)
try:
    c = CropModel.load_from_checkpoint(CLS, map_location="cpu")
    labs = list((getattr(c, "label_dict", {}) or {}).keys())
    print(f"OK classification loaded; n_labels = {len(labs)}", flush=True)
    seals = [l for l in labs if "Phoc" in l or "Halichoerus" in l]
    print("seal labels present in classifier:", seals, flush=True)
except Exception as e:
    print(f"FAIL classification: {type(e).__name__}: {str(e)[:500]}", flush=True)
