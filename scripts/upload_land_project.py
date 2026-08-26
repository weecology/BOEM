"""Create the land/water screening project in Label Studio and upload mined frames.

Run scripts/mine_land_examples.py first to produce the manifest, then:

    uv run python scripts/upload_land_project.py --dry-run
    uv run python scripts/upload_land_project.py

This is a whole-frame classification project, not a detection one, so it does not use
`upload_to_label_studio` (which hardcodes the RectangleLabels config). It calls
connect/upload/import directly with the classification config below.

The task order is shuffled by the miner and the model's own guess is deliberately not
shown, so annotators cannot anchor on it -- the point of the exercise is an
independent label.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import label_studio as ls_mod
from src.label_studio import get_api_key

MANIFEST = Path("/blue/ewhite/b.weinstein/BOEM/annotations/land_screen/manifest.csv")
IMAGERY = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
PROJECT_NAME = "Bureau of Ocean Energy Management - Land Screen"

# Whole-frame, single-choice. Hotkeys matter: annotators run through hundreds of these,
# and the four categories are deliberate. "Mixed" exists so a shoreline is not forced
# into a binary, and "Unusable" so truncated frames (there is at least one half-black
# file in JPG_20241219_120500) do not get labelled as water.
LABEL_CONFIG = """<View>
  <Header value="Is this frame over land or water?"/>
  <Text name="guide" value="Water = open water only, ANY colour or texture: blue, green, brown, grey glint, waves, whitecaps, foam, wakes. Texture alone is NOT land. Land = any solid ground: beach, dune, marsh, forest, grass, buildings, roads, docks. Mixed = both present, e.g. a shoreline crossing the frame. Unusable = black, truncated, or too blurred to judge."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" width="100%"/>
  <Choices name="surface" toName="image" choice="single-radio" required="true" showInLine="true">
    <Choice value="Water" hotkey="1" background="#1f77b4"/>
    <Choice value="Land" hotkey="2" background="#8c564b"/>
    <Choice value="Mixed" hotkey="3" background="#ff7f0e"/>
    <Choice value="Unusable" hotkey="4" background="#7f7f7f"/>
  </Choices>
</View>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST)
    images = [str(IMAGERY / r.flight / r.image) for r in manifest.itertuples()]
    missing = [p for p in images if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest frames not on disk, e.g. {missing[0]}")

    print(f"{len(images)} frames from {manifest.flight.nunique()} flights -> '{PROJECT_NAME}'")
    print(manifest.rule_says.value_counts().to_string())
    if args.dry_run:
        print("\nDRY-RUN: no project created, nothing uploaded.")
        print(LABEL_CONFIG)
        return

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No Label Studio API key found in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None):
        cfg = compose(config_name="boem_config")
    ls_cfg = cfg.annotation.label_studio

    project = ls_mod.connect_to_label_studio(
        url=ls_cfg.url, project_name=PROJECT_NAME, label_config=LABEL_CONFIG)
    sftp_client = ls_mod.create_sftp_client(
        user=cfg.server.user, host=cfg.server.host, key_filename=cfg.server.key_filename)
    ls_mod.upload_images(sftp_client=sftp_client, images=images, folder_name=ls_cfg.folder_name)
    ls_mod.import_image_tasks(
        label_studio_project=project, image_names=images, local_image_dir=str(IMAGERY))
    print(f"\nUploaded {len(images)} frames to '{PROJECT_NAME}'.")


if __name__ == "__main__":
    main()
