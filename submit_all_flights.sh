#!/bin/bash
#
# Submit BOEM prediction jobs for each directory containing images.
#
# Usage:
#   ./submit_all_flights.sh [OPTIONS] [START_DIR] [PATH_GLOB]
#
# Options:
#   --serial        run all directories in one GPU job (sequential, avoids queue churn)
#   --max-depth N   limit directory recursion depth
#   --time HH:MM:SS wall-clock limit (default 48:00:00; increase for --serial runs)
#   --dry-run       list matching dirs without submitting
#   --help
#
# PATH_GLOB is matched against the full path; quote to avoid shell expansion (e.g. '*Gulf*').
# Only directories with at least one .jpg/.jpeg are submitted.

IMAGERY_ROOT="/blue/ewhite/b.weinstein/BOEM/imagery"
SLEEP_SEC=3
MAX_DEPTH=""
DRY_RUN=false
SERIAL=false
TIME_LIMIT="48:00:00"
START_DIR=""
PATH_GLOB=""

usage() { grep '^#' "$0" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)   usage; exit 0 ;;
    --dry-run)   DRY_RUN=true; shift ;;
    --serial)    SERIAL=true; shift ;;
    --max-depth) MAX_DEPTH="$2"; shift 2 ;;
    --time)      TIME_LIMIT="$2"; shift 2 ;;
    -*)          echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      if   [[ -z "$START_DIR" ]]; then START_DIR="$1"
      elif [[ -z "$PATH_GLOB" ]]; then PATH_GLOB="$1"
      fi
      shift ;;
  esac
done

START_DIR="${START_DIR:-$IMAGERY_ROOT}"
[[ -d "$START_DIR" ]] || { echo "Error: not a directory: $START_DIR" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

has_jpg() {
  [[ -n $(find "$1" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print -quit 2>/dev/null) ]]
}

slurm_header() {
  cat <<HEADER
#!/bin/bash
#SBATCH --job-name=BOEM
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80GB
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --output=/home/b.weinstein/logs/BOEM_%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM_%j.err

ulimit -c 0
module load ffmpeg
HEADER
}

# pool_limit=null scores every image in the flight rather than a random subsample.
# batch_size must stay 1: peak memory is ~6.4 GB per image in the batch, so the 4 this
# used to pass needs ~25.5 GB and does not fit the 24 GB L4 requested above. It is also
# the fastest setting — see the benchmark table in boem_conf/boem_config.yaml.
PYTHON_CMD="uv run python main.py check_annotations=True active_learning.pool_limit=null predict.batch_size=1 debug=False"

# Coarse throughput accounting: one CSV line per flight, wall time over image count and
# bytes, so we can answer "how long per TB per GPU". Deliberately loose — it times the
# whole pipeline (detection + classification + report), not just inference, and the size
# walk is a single find over the directory.
THROUGHPUT_CSV="/blue/ewhite/b.weinstein/BOEM/throughput.csv"

emit_timed_run() {
  local folder="$1"
  cat <<TIMED
echo "Processing: ${folder}"
read n_images n_bytes < <(find "${folder}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -printf '%s\n' | awk '{n++; b+=\$1} END {print n+0, b+0}')
t_start=\$SECONDS
${PYTHON_CMD} image_dir="${folder}"
rc=\$?
elapsed=\$(( SECONDS - t_start ))
awk -v f="\$(basename ${folder})" -v j="\$SLURM_JOB_ID" -v n="\$n_images" -v b="\$n_bytes" -v s="\$elapsed" -v rc="\$rc" \
  'BEGIN { tb = b / 1099511627776; printf "%s,%s,%d,%d,%d,%.4f,%.3f,%.1f,%d\n", f, j, n, b, s, s/3600, (s>0 ? n/s : 0), (tb>0 ? (s/3600)/tb : 0), rc }' >> "${THROUGHPUT_CSV}"
TIMED
}

submit_parallel() {
  local folder="$1"
  sbatch < <(
    slurm_header
    emit_timed_run "$folder"
  )
  sleep "$SLEEP_SEC"
}

submit_serial() {
  sbatch < <(
    slurm_header
    for folder in "$@"; do
      emit_timed_run "$folder"
    done
  )
}

# ---------------------------------------------------------------------------
# Collect matching directories
# ---------------------------------------------------------------------------

find_args=( "$START_DIR" -type d -mindepth 1 )
[[ -n "$MAX_DEPTH" ]] && find_args+=( -maxdepth "$MAX_DEPTH" )

folders=()
while IFS= read -r -d '' folder; do
  [[ -n "$PATH_GLOB" ]] && ! [[ "$folder" == $PATH_GLOB ]] && continue
  has_jpg "$folder" || continue
  folders+=( "$folder" )
done < <(find "${find_args[@]}" -print0 | sort -z)

count=${#folders[@]}

# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

if [[ "$DRY_RUN" == true ]]; then
  printf '%s\n' "${folders[@]}"
  if [[ "$SERIAL" == true ]]; then
    echo "Dry run: $count directories → 1 serial job"
  else
    echo "Dry run: $count directories → $count parallel jobs"
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

if [[ "$SERIAL" == true ]]; then
  echo "Submitting 1 serial job for $count directories..."
  submit_serial "${folders[@]}"
else
  echo "Submitting $count parallel jobs..."
  for folder in "${folders[@]}"; do
    echo "  $folder"
    submit_parallel "$folder"
  done
  echo "Done: submitted $count jobs."
fi
