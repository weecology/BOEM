"""Upload the current BOEM detection model to the Hugging Face Hub as a PR.

Follows https://deepforest.readthedocs.io/en/stable/development/contributing.html#upload-to-hugging-face-hub
Auth: reads token from $HF_HOME/token (set via `huggingface-cli login`) or $HF_TOKEN.

Usage: python scripts/upload_detection_to_hf.py OUT_DIR [--push]
Without --push the files are only written to OUT_DIR for inspection.
"""
import json
import shutil
import sys
from pathlib import Path

from deepforest import main
from huggingface_hub import HfApi

CHECKPOINT = (
    "/blue/ewhite/b.weinstein/BOEM/training/checkpoints/"
    "1968e962113f4c57bc6bf5f07fae1b25/epoch29-val_cls0.0899.ckpt"
)
REPO_ID = "weecology/deepforest-marine-biodiversity"
LABEL_DICT = {"Object": 0}
MODEL_CARD = Path(__file__).with_name("hf_model_card.md")

out_dir = Path(sys.argv[1])
model = main.deepforest.load_from_checkpoint(CHECKPOINT)
model.model.save_pretrained(out_dir)

# save_pretrained serialises RetinaNetHub's init kwargs, which still carry the
# deepforest-tree default {"Tree": 0}; load_model reads label_dict from this file.
config_path = out_dir / "config.json"
config = json.loads(config_path.read_text())
config["label_dict"] = LABEL_DICT
config_path.write_text(json.dumps(config, indent=2) + "\n")
shutil.copy(MODEL_CARD, out_dir / "README.md")

if "--push" in sys.argv:
    commit = HfApi().upload_folder(
        repo_id=REPO_ID,
        folder_path=out_dir,
        create_pr=True,
        commit_message="Update to BOEM detector 1968e962/epoch29 and add model card",
    )
    print(commit.pr_url)
