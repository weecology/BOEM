#!/bin/bash
# One-command SLURM route for USGS annotation preparation.
# 1) Build manifest of stale/missing detection crop CSVs
# 2) Run shard array to refresh only those crops
# 3) Run final prepare_USGS stages 1-3 (skip Stage 0)

set -euo pipefail

cd "$(dirname "$0")"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="tmp/prepare_annotations_${RUN_ID}"
MANIFEST="${WORK_DIR}/manifest.csv"
SHARD_DIR="${WORK_DIR}/shards"

mkdir -p "$SHARD_DIR"

echo "Building manifest..."
uv run python scripts/prepare_USGS.py --write-detection-refresh-manifest "$MANIFEST"

N_ROWS="$(uv run python - <<'PY' "$MANIFEST"
import pandas as pd
import sys
print(len(pd.read_csv(sys.argv[1])))
PY
)"

if [[ "$N_ROWS" -eq 0 ]]; then
  echo "No stale crops found. Running final prepare only."
  sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=prep_ann_final
#SBATCH --account=ewhite
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/b.weinstein/logs/prep_ann_final_%j.out
#SBATCH --error=/home/b.weinstein/logs/prep_ann_final_%j.err
cd /blue/ewhite/b.weinstein/BOEM || exit 1
uv run python scripts/prepare_USGS.py --no-generate-detection-crops
EOF
  exit 0
fi

N_SHARDS="$(uv run python - <<'PY' "$MANIFEST" "$SHARD_DIR"
import math
import pandas as pd
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
shard_size = 200
df = pd.read_csv(manifest)
n = len(df)
n_shards = math.ceil(n / shard_size)
for i in range(n_shards):
    start = i * shard_size
    end = min((i + 1) * shard_size, n)
    df.iloc[start:end].to_csv(shard_dir / f"shard_{i:05d}.csv", index=False)
print(n_shards)
PY
)"

ARRAY_JOB_ID="$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=prep_ann_refresh
#SBATCH --account=ewhite
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=03:00:00
#SBATCH --array=0-$((N_SHARDS-1))%80
#SBATCH --output=/home/b.weinstein/logs/prep_ann_refresh_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/prep_ann_refresh_%A_%a.err
cd /blue/ewhite/b.weinstein/BOEM || exit 1
SHARD_FILE="${SHARD_DIR}/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}").csv"
uv run python scripts/prepare_USGS.py --process-detection-refresh-manifest "${SHARD_FILE}"
EOF
)"

FINAL_JOB_ID="$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=prep_ann_final
#SBATCH --account=ewhite
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --dependency=afterok:${ARRAY_JOB_ID}
#SBATCH --output=/home/b.weinstein/logs/prep_ann_final_%j.out
#SBATCH --error=/home/b.weinstein/logs/prep_ann_final_%j.err
cd /blue/ewhite/b.weinstein/BOEM || exit 1
uv run python scripts/prepare_USGS.py --no-generate-detection-crops
EOF
)"

echo "Submitted refresh array job: ${ARRAY_JOB_ID}"
echo "Submitted final prepare job: ${FINAL_JOB_ID}"
