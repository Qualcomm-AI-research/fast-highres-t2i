#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Simple PixelRush evaluation workflow.
#
# Edit the variables below, then run:
#   bash eval/run_eval_workflow.sh

set -euo pipefail

PROMPT_FILE="eval/eval_prompt.txt"
OUTPUT_DIR="eval_outputs/pixelrush_sdxl"
REF_STATS="eval/reference_stats.npz"

NUM_IMAGES=1000
NUM_GPUS=8
MODEL="sdxl-turbo"

# Choose one:
#   all      Generate 1K, generate 2K/4K, then evaluate.
#   base     Generate only 1K images.
#   upscale  Generate only 2K/4K images from existing image_1k.
#   eval     Evaluate existing image_1k/image_2k/image_4k.
STAGE="all"

# Upscale options.
USE_TEMPORAL_SCALING=true
OVERWRITE=true

count_images() {
  local image_dir="$1"
  if [[ ! -d "${image_dir}" ]]; then
    echo 0
    return
  fi

  find "${image_dir}" -type f \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.webp" \) \
    | wc -l \
    | tr -d " "
}

check_image_count() {
  local scale_dir="$1"
  local image_dir="${OUTPUT_DIR}/${scale_dir}"
  local count
  count="$(count_images "${image_dir}")"

  echo "${scale_dir}: ${count}/${EXPECTED_IMAGES} images"

  if [[ "${count}" -lt "${EXPECTED_IMAGES}" ]]; then
    echo "ERROR: ${image_dir} is incomplete."
    echo "Expected ${EXPECTED_IMAGES} images, found ${count}."
    exit 1
  fi
}

generation_args() {
  local stage="$1"

  local args=(
    eval/generate_pixelrush_eval.py
    --prompt_file "${PROMPT_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --num_images "${NUM_IMAGES}"
    --stage "${stage}"
    --model "${MODEL}"
  )

  if [[ "${USE_TEMPORAL_SCALING}" == true ]]; then
    args+=(--use_temporal_scaling)
  else
    args+=(--no_temporal_scaling)
  fi

  if [[ "${OVERWRITE}" == true ]]; then
    args+=(--overwrite)
  fi

  printf '%s\n' "${args[@]}"
}

run_generation_stage() {
  local stage="$1"

  if [[ "${NUM_GPUS}" -le 1 ]]; then
    mapfile -t args < <(generation_args "${stage}")
    python "${args[@]}" --rank 0 --world_size 1
  else
    for RANK in $(seq 0 "$((NUM_GPUS - 1))"); do
      mapfile -t args < <(generation_args "${stage}")
      CUDA_VISIBLE_DEVICES="${RANK}" python "${args[@]}" \
        --device cuda:0 \
        --rank "${RANK}" \
        --world_size "${NUM_GPUS}" &
    done
    wait
  fi
}

run_eval() {
  for scale_dir in image_1k image_2k image_4k; do
    python eval/evaluate.py \
      --image_dir "${OUTPUT_DIR}/${scale_dir}" \
      --ref_stats "${REF_STATS}" \
      --output_json "${OUTPUT_DIR}/${scale_dir}_metrics.json" \
      --batch_size 32
  done
}

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Prompt file not found: ${PROMPT_FILE}"
  exit 1
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "eval" ]]; then
  if [[ ! -f "${REF_STATS}" ]]; then
    echo "Reference stats not found: ${REF_STATS}"
    exit 1
  fi
fi

PROMPT_COUNT="$(grep -cv '^[[:space:]]*$' "${PROMPT_FILE}")"
EXPECTED_IMAGES="${NUM_IMAGES}"
if [[ "${PROMPT_COUNT}" -lt "${NUM_IMAGES}" ]]; then
  EXPECTED_IMAGES="${PROMPT_COUNT}"
fi

echo "Prompt file:     ${PROMPT_FILE}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Ref stats:       ${REF_STATS}"
echo "Num images:      ${NUM_IMAGES}"
echo "Expected images: ${EXPECTED_IMAGES}"
echo "Num GPUs:        ${NUM_GPUS}"
echo "Model:           ${MODEL}"
echo "Stage:           ${STAGE}"
echo "Temporal scale:  ${USE_TEMPORAL_SCALING}"
echo "Overwrite:       ${OVERWRITE}"

mkdir -p "${OUTPUT_DIR}"

case "${STAGE}" in
  all)
    echo
    echo "Generating 1K images..."
    run_generation_stage "base"
    check_image_count "image_1k"

    echo
    echo "Generating 2K and 4K images..."
    run_generation_stage "upscale"
    check_image_count "image_2k"
    check_image_count "image_4k"

    echo
    echo "Evaluating images..."
    run_eval
    ;;

  base)
    echo
    echo "Generating 1K images..."
    run_generation_stage "base"
    check_image_count "image_1k"
    ;;

  upscale)
    echo
    echo "Generating 2K and 4K images from existing image_1k..."
    check_image_count "image_1k"
    run_generation_stage "upscale"
    check_image_count "image_2k"
    check_image_count "image_4k"
    ;;

  eval)
    echo
    echo "Evaluating images..."
    check_image_count "image_1k"
    check_image_count "image_2k"
    check_image_count "image_4k"
    run_eval
    ;;

  *)
    echo "Invalid STAGE: ${STAGE}"
    echo "Use one of: all, base, upscale, eval"
    exit 1
    ;;
esac

echo
echo "Done: ${OUTPUT_DIR}"
