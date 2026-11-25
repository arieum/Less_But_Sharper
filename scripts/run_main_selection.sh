#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_TAG="${MODEL_TAG:-llama2-7b}"
DATA_DIR="${DATA_DIR:-${PACKAGE_ROOT}/data}"
WORK_DIR="${WORK_DIR:-${PACKAGE_ROOT}/work/${MODEL_TAG}}"
K="${K:-30}"
LAMBDA="${LAMBDA:-0.6}"
NUM_TOKENS="${NUM_TOKENS:-30000}"
DEVICE="${DEVICE:-cuda}"
CANDIDATES="${CANDIDATES:-${WORK_DIR}/${MODEL_TAG}.neuron.pth}"
META="${META:-${WORK_DIR}/${MODEL_TAG}.meta.pth}"
LANGUAGES=(en zh fr es vi id ja)

mkdir -p "${WORK_DIR}/signatures" "${WORK_DIR}/masks"

for lang in "${LANGUAGES[@]}"; do
  python "${PACKAGE_ROOT}/code/selection/collect_token_signatures.py" \
    --model_tag "${MODEL_TAG}" --lang "${lang}" \
    --in_neuron "${CANDIDATES}" --num_tokens "${NUM_TOKENS}" \
    --data_dir "${DATA_DIR}" --out_dir "${WORK_DIR}/signatures"
done

python "${PACKAGE_ROOT}/code/selection/select_topk.py" \
  --input_neuron "${CANDIDATES}" --meta_path "${META}" --K "${K}" \
  --out_neuron "${WORK_DIR}/masks/topk_k${K}.pth"

python "${PACKAGE_ROOT}/code/selection/select_mmr.py" \
  --sig_dir "${WORK_DIR}/signatures" --model_tag "${MODEL_TAG}" \
  --meta_path "${META}" --num_tokens "${NUM_TOKENS}" --K "${K}" \
  --lambda_ "${LAMBDA}" --device "${DEVICE}" \
  --out_neuron "${WORK_DIR}/masks/mmr_k${K}_lambda${LAMBDA}.pth"
