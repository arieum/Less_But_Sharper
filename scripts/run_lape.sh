#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_TAG="${MODEL_TAG:-llama2-7b}"
DATA_DIR="${DATA_DIR:-${PACKAGE_ROOT}/data}"
WORK_DIR="${WORK_DIR:-${PACKAGE_ROOT}/work/${MODEL_TAG}}"
MAX_TOKENS="${MAX_TOKENS:-}"
LANGUAGES=(en zh fr es vi id ja)

mkdir -p "${DATA_DIR}" "${WORK_DIR}"

for lang in "${LANGUAGES[@]}"; do
  extra_args=()
  if [[ -n "${MAX_TOKENS}" ]]; then
    extra_args+=(--max_tokens "${MAX_TOKENS}")
  fi
  python "${PACKAGE_ROOT}/code/lape/collect_activations.py" \
    --model_tag "${MODEL_TAG}" --lang "${lang}" \
    --data_dir "${DATA_DIR}" "${extra_args[@]}"
done

python "${PACKAGE_ROOT}/code/lape/extract_lape_candidates.py" \
  --model_tag "${MODEL_TAG}" \
  --data_dir "${DATA_DIR}" \
  --top_rate 0.01 \
  --out_prefix "${WORK_DIR}/${MODEL_TAG}"
