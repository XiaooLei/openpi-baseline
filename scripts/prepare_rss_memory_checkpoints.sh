#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

SRC_ROOT=${SRC_ROOT:-/inspire/qb-ilm/project/gjjproject/public/xhc/checkpoints/2nd-submit}
DST_ROOT=${DST_ROOT:-/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints/2nd-submit}

prepare_one() {
  local name=$1
  local task_slug=$2
  local archive=$3
  local dst="$DST_ROOT/$name"
  local tar_path="$SRC_ROOT/$archive"
  local params_path="$dst/params"
  local norm_stats_path="$dst/assets/$task_slug/expert-success-hil-suffix-mix-data/norm_stats.json"

  if [[ ! -e "$tar_path" ]]; then
    echo "Missing checkpoint archive: $tar_path" >&2
    exit 1
  fi

  if [[ -e "$params_path/_METADATA" && -e "$norm_stats_path" ]]; then
    echo "Ready: $name"
    return
  fi

  echo "Extracting $archive -> $dst"
  mkdir -p "$dst"
  tar -xzf "$tar_path" -C "$dst"

  if [[ ! -e "$params_path/_METADATA" ]]; then
    echo "Missing extracted params metadata: $params_path/_METADATA" >&2
    exit 1
  fi
  if [[ ! -e "$norm_stats_path" ]]; then
    echo "Missing extracted norm stats: $norm_stats_path" >&2
    exit 1
  fi

  echo "Ready: $name"
}

prepare_one "mouse_80k" "insert-mouse-battery" "mouse_80k_assets_params.tar.gz"
prepare_one "hanoi_200k" "tower-of-hanoi-game" "hanoi_200k_assets_params.tar.gz"
prepare_one "water_120k" "seal-water-bottle-cap" "water_120k_assets_params.tar.gz"

echo "All RSS memory checkpoints are ready under $DST_ROOT"
