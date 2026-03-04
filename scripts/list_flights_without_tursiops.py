#!/usr/bin/env python3
"""
List the largest N flights (by directory size) that have a prediction cache
but no Tursiops truncatus predictions. Use to identify candidate flights to
move/archive to free space without affecting Label Studio (no dolphin predictions).
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd

IMAGERY_ROOT = Path("/blue/ewhite/b.weinstein/BOEM/imagery")
TARGET_SPECIES = "Tursiops truncatus"
TOP_N = 5


def has_tursiops(csv_path: Path):
    """True if CSV has any Tursiops truncatus in cropmodel_label or hcast_species; None if unreadable."""
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in ("cropmodel_label", "hcast_species"))
    except Exception:
        return None
    for col in ("cropmodel_label", "hcast_species"):
        if col in df.columns and df[col].astype(str).str.contains(TARGET_SPECIES, na=False).any():
            return True
    return False


def flight_size_mb(flight_path: Path) -> int:
    """Size of directory in MB (du -sm)."""
    try:
        out = subprocess.run(
            ["du", "-sm", str(flight_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if out.returncode == 0:
            return int(out.stdout.split()[0])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return 0


def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else TOP_N
    imagery = Path(sys.argv[2]) if len(sys.argv) > 2 else IMAGERY_ROOT

    candidates = []
    for flight_dir in sorted(imagery.iterdir()):
        if not flight_dir.is_dir():
            continue
        pred_file = flight_dir / ".prediction_cache" / "pool_predictions.csv"
        if not pred_file.exists():
            continue
        if has_tursiops(pred_file):
            continue
        size_mb = flight_size_mb(flight_dir)
        candidates.append((flight_dir.name, size_mb))

    candidates.sort(key=lambda x: -x[1])
    top = candidates[:top_n]

    print(f"Largest {len(top)} flight(s) with prediction cache and no '{TARGET_SPECIES}' predictions:\n")
    for name, size_mb in top:
        print(f"  {size_mb:>10,} MB   {name}")
    print(f"\nTotal flights with cache and no {TARGET_SPECIES}: {len(candidates)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
