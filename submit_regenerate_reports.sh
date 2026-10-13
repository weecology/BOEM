#!/bin/bash
#
# Bring flight reports up to date with Label Studio without re-running any model.
#
# Usage:
#   ./submit_regenerate_reports.sh '*JPG_202607*'
#
# 1. Pulls every annotated task from Label Studio (tasks stay on the server) and backs
#    the new CSVs up into annotations_backup/ -- once, here, not once per flight.
# 2. Submits ONE CPU job that runs main.py report_only=True on each matching flight:
#    predictions come from <flight>/.prediction_cache, labels from the annotation CSVs.
#    A flight whose cache was not written by the configured detector and classifier
#    fails there and needs the full pipeline (submit_all_flights.sh) instead.
#
# Log the job in JOB_LEDGER.md.

set -euo pipefail
GLOB="$1"
IMAGERY_ROOT="/blue/ewhite/b.weinstein/BOEM/imagery"
cd "$(dirname "$0")"

uv run python scripts/download_annotations.py
uv run python scripts/backup_annotations.py

sbatch --parsable <<SBATCH
#!/bin/bash
#SBATCH --job-name=BOEM_reports
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM_reports_%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM_reports_%j.err

cd $(pwd)
for flight in ${IMAGERY_ROOT}/${GLOB}; do
  [[ -d "\$flight/.prediction_cache" ]] || continue
  echo "=== report_only: \$flight"
  uv run python main.py report_only=True check_annotations=False debug=False \
    flythrough_video.enabled=False image_dir="\$flight" || echo "FAILED: \$flight"
done
SBATCH
