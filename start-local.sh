#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REL_PATH="${MODEL_REL_PATH:-models/all-MiniLM-L6-V2}"
export MODEL_REL_PATH

if [[ "${MODEL_REL_PATH}" = /* ]]; then
  MODEL_DIR="${MODEL_REL_PATH}"
else
  MODEL_DIR="${ROOT_DIR}/${MODEL_REL_PATH}"
fi

export HF_MODELS_DIR="${HF_MODELS_DIR:-${ROOT_DIR}/models}"

echo "🚀 Starting DFX HF Models Node"
echo "   NATS URL : ${NATS_URL:-nats://localhost:4222}"
echo "   Model dir: ${MODEL_DIR}"

# Ensure local SentenceTransformer model is available
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "📦 Local MiniLM model not found. Downloading..."
    "${ROOT_DIR}/scripts/download.sh" sentence-transformers/all-MiniLM-L6-V2 "${MODEL_DIR}"
fi

# Make sure src/ and dfx/ are discoverable
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/dfx:${PYTHONPATH:-}"

# Sensible defaults for node metadata
export NODE_NAME="${NODE_NAME:-dfx-hf-models-node}"
export NATS_URL="${NATS_URL:-nats://localhost:4222}"
export STREAM_NAME="${STREAM_NAME:-droq-stream}"
export RELOAD="${RELOAD:-true}"

uv run python -m node.main


