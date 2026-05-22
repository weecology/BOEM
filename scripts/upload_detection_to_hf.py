"""Upload the current BOEM detection model to the Hugging Face Hub as a PR.

Follows https://deepforest.readthedocs.io/en/stable/development/contributing.html#upload-to-hugging-face-hub
Auth: reads token from $HF_HOME/token (set via `huggingface-cli login`) or $HF_TOKEN.
"""
from deepforest import main

CHECKPOINT = (
    "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/"
    "checkpoints/a1c5649615f94b33a4e0dc2a721009fc.pl"
)
REPO_ID = "weecology/deepforest-marine-biodiversity"
LABEL_DICT = {"Object": 0}

model = main.deepforest.load_from_checkpoint(CHECKPOINT)
model.label_dict = LABEL_DICT

url = model.model.push_to_hub(
    REPO_ID,
    create_pr=True,
    commit_message="Add BOEM detection model checkpoint",
)
print(url)
