#!/bin/bash
#
# Submit BOEM jobs for each directory under imagery (recursively).
#
# Usage:
#   ./submit_all_flights.sh                          # all dirs under default imagery root
#   ./submit_all_flights.sh /path/to/start           # only dirs under this path
#   ./submit_all_flights.sh /path/to/start '*Gulf*'  # only dirs whose path matches glob
#
# Options (must appear before positional args):
#   --max-depth N   limit recursion depth (1 = immediate children only)
#   --dry-run       print dirs that would be submitted, do not sbatch
#   --help          show this usage
#
# Glob is matched against the full path; quote it to avoid shell expansion (e.g. '*Gulf*').
#
# Only directories containing at least one .jpg/.jpeg file (any case) are submitted.
#

has_jpg() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -quit 2>/dev/null
}

IMAGERY_ROOT="/blue/ewhite/b.weinstein/BOEM/imagery"
SLEEP_SEC=3
MAX_DEPTH=""
DRY_RUN=false
START_DIR=""
PATH_GLOB=""

usage() {
  echo "Usage: $0 [OPTIONS] [START_DIR] [PATH_GLOB]"
  echo "  START_DIR   default: $IMAGERY_ROOT"
  echo "  PATH_GLOB   e.g. '*Gulf*' to limit to paths containing Gulf"
  echo "  Options: --max-depth N, --dry-run, --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --max-depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z "$START_DIR" ]]; then
        START_DIR="$1"
      elif [[ -z "$PATH_GLOB" ]]; then
        PATH_GLOB="$1"
      fi
      shift
      ;;
  esac
done

START_DIR="${START_DIR:-$IMAGERY_ROOT}"

if [[ ! -d "$START_DIR" ]]; then
  echo "Error: START_DIR is not a directory: $START_DIR" >&2
  exit 1
fi

# Build find command: all dirs under START_DIR, optional max depth
find_args=( "$START_DIR" -type d -mindepth 1 )
if [[ -n "$MAX_DEPTH" ]]; then
  find_args+=( -maxdepth "$MAX_DEPTH" )
fi

count=0
while IFS= read -r -d '' folder; do
  if [[ -n "$PATH_GLOB" ]]; then
    if ! [[ "$folder" == $PATH_GLOB ]]; then
      continue
    fi
  fi
  if ! has_jpg "$folder"; then
    continue
  fi
  count=$((count + 1))
  echo "Submitting job with image_dir: $folder"
  if [[ "$DRY_RUN" == true ]]; then
    continue
  fi
  sbatch --export=ALL,IMAGE_DIR="$folder" <<EOF
#!/bin/bash
#SBATCH --job-name=BOEM
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --output=/home/b.weinstein/logs/BOEM_%j.out
#SBATCH --error=/home/b.weinstein/logs/BOEM_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

uv run python main.py image_dir=\$IMAGE_DIR check_annotations=True active_learning.pool_limit=1000 debug=False
EOF
  sleep "$SLEEP_SEC"
done < <(find "${find_args[@]}" -print0 | sort -z)

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run: would submit $count jobs."
else
  echo "Submitted $count jobs."
fi
