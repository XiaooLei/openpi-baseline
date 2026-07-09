#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_DIR=${BASE_DIR:-/inspire/qb-ilm/project/gjjproject/public/xl}
OUT=${OUT:-"$BASE_DIR/rss_memory_training_data_$(date '+%Y%m%d-%H%M%S').tar"}

required_paths=(
  "data/rss_challenge/raw/insert-mouse-battery/expert-success-hil-suffix-mix-data"
  "data/rss_challenge/raw/seal-water-bottle-cap/expert-success-hil-suffix-mix-data"
  "data/rss_challenge/raw/tower-of-hanoi-game/expert-success-hil-suffix-mix-data"
  "data/rss_challenge/recap/phase2/insert_mouse_battery_hil_split/train"
  "data/rss_challenge/recap/phase2/seal_water_bottle_cap_hil_split/train"
  "data/rss_challenge/recap/phase2/tower_of_hanoi_game_hil_split/train"
  "data/baseline_checkpoints/2nd-submit/mouse_80k"
  "data/baseline_checkpoints/2nd-submit/hanoi_200k"
  "data/baseline_checkpoints/2nd-submit/water_120k"
  "data/baseline_checkpoints/pi05_seal-water-bottle-cap/199999"
  "openpi-baseline/.cache/openpi/big_vision/paligemma_tokenizer.model"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$BASE_DIR/$path" ]]; then
    echo "Missing required path for data bundle: $BASE_DIR/$path" >&2
    exit 1
  fi
done

if [[ -e "$OUT" ]]; then
  echo "Output tar already exists: $OUT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

echo "Creating data tar: $OUT"
printf '%s\n' "${required_paths[@]}" | tar -C "$BASE_DIR" -cf "$OUT" --files-from=- --totals
echo "Created data tar: $OUT"
du -sh "$OUT"
