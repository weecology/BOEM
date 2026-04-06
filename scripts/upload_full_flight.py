"""Upload all detections from a single flight to a dedicated Label Studio project.

Loads existing annotations (train/eval/review) and fills remaining images with
model predictions, then uploads everything to a per-flight Label Studio project
for video annotation cleanup.

Usage:
    uv run python scripts/upload_full_flight.py JPG_20241107_135800
    uv run python scripts/upload_full_flight.py /full/path/to/imagery/JPG_20241107_135800
    uv run python scripts/upload_full_flight.py JPG_20241107_135800 --project-name "My Flight Review"
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGERY_BASE = "/blue/ewhite/b.weinstein/BOEM/imagery"

load_dotenv(PROJECT_ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(
        description="Upload all flight detections to a Label Studio project"
    )
    parser.add_argument(
        "flight_name",
        help="Flight directory name (e.g. JPG_20241107_135800) or full path to imagery dir",
    )
    parser.add_argument(
        "--project-name",
        default=None,
        help="Override Label Studio project name (default: 'BOEM - Full Flight - {flight_name}')",
    )
    args = parser.parse_args()

    # Accept either a bare flight name or a full path
    if os.path.isabs(args.flight_name):
        image_dir = args.flight_name
    else:
        image_dir = os.path.join(IMAGERY_BASE, args.flight_name)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Set up Label Studio API key from config file
    from src.label_studio import get_api_key
    api_key = get_api_key()
    if api_key is None:
        raise RuntimeError("No Label Studio API key found in .label_studio.config")
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "boem_conf"), version_base=None
    ):
        cfg = compose(
            config_name="boem_config",
            overrides=[f"image_dir={image_dir}", "check_annotations=true"],
        )

    print(f"Flight: {os.path.basename(image_dir)}")
    print(f"Image dir: {image_dir}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    from src.pipeline import Pipeline
    pipeline = Pipeline(cfg=cfg)
    pipeline.upload_full_flight(project_name=args.project_name)


if __name__ == "__main__":
    main()
