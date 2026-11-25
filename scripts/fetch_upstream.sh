#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${PACKAGE_ROOT}/external/Language-Specific-Neurons"
UPSTREAM_URL="https://github.com/RUCAIBox/Language-Specific-Neurons.git"
UPSTREAM_COMMIT="fdeca23acd53d7a4de5ca3f7a1b66ef94d7e19a7"

mkdir -p "${PACKAGE_ROOT}/external" "${PACKAGE_ROOT}/data"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
fi

git -C "${UPSTREAM_DIR}" fetch origin
git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"

PROMPT_SOURCE="${UPSTREAM_DIR}/data/mvicuna"
PROMPT_LINK="${PACKAGE_ROOT}/data/mvicuna"
if [[ -e "${PROMPT_LINK}" && ! -L "${PROMPT_LINK}" ]]; then
  echo "Refusing to replace existing path: ${PROMPT_LINK}" >&2
  exit 1
fi
ln -sfn "${PROMPT_SOURCE}" "${PROMPT_LINK}"

echo "Upstream checkout: ${UPSTREAM_DIR} @ ${UPSTREAM_COMMIT}"
echo "mVicuna prompts:   ${PROMPT_LINK} -> ${PROMPT_SOURCE}"
