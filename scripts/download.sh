#!/usr/bin/env bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install via https://github.com/astral-sh/uv" >&2
    exit 1
fi

if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
    echo "Usage: $0 <model-name> [destination]" >&2
    exit 1
fi

MODEL_NAME="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"
mkdir -p "$MODELS_DIR"

DEST_DIR="${2:-${MODELS_DIR}/${MODEL_NAME##*/}}"

mkdir -p "$DEST_DIR"

echo "Downloading ${MODEL_NAME} into ${DEST_DIR}..."
uv run huggingface-cli download "$MODEL_NAME" --local-dir "$DEST_DIR" --local-dir-use-symlinks False

echo "Model downloaded to ${DEST_DIR}"