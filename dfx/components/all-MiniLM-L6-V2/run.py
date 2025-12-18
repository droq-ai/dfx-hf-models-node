#!/usr/bin/env python3
"""Utility runner for the locally stored all-MiniLM-L6-V2 embeddings model."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "sentence-transformers must be installed to run this script. "
        "Install it with `uv pip install sentence-transformers`."
    ) from exc


MODEL_NAME = "all-MiniLM-L6-V2"
ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT_DIR / "models" / MODEL_NAME
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-V2"
DOWNLOAD_SCRIPT = ROOT_DIR / "scripts" / "download.sh"


def ensure_model(model_path: Path = MODEL_PATH) -> Path:
    """Ensure the local MiniLM model assets exist."""
    if model_path.exists():
        return model_path

    if not DOWNLOAD_SCRIPT.exists():
        raise FileNotFoundError(f"Download script not found at {DOWNLOAD_SCRIPT}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(DOWNLOAD_SCRIPT), HF_MODEL_ID, str(model_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Failed to download {HF_MODEL_ID}: {detail}")

    if not model_path.exists():
        raise RuntimeError(f"Download reported success but assets missing at {model_path}")

    return model_path


def load_model(model_path: Path = MODEL_PATH) -> SentenceTransformer:
    """Load the embedding model from the local models directory."""
    model_path = ensure_model(model_path)
    return SentenceTransformer(str(model_path))


def run(texts: Iterable[str]) -> list[list[float]]:
    """Generate embeddings for the provided texts."""
    texts_list = [text for text in texts if text.strip()]
    if not texts_list:
        raise ValueError("At least one non-empty text input is required.")

    model = load_model()
    embeddings = model.encode(texts_list)
    return [embedding.tolist() for embedding in embeddings]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the all-MiniLM-L6-V2 SentenceTransformer from the local cache."
    )
    parser.add_argument(
        "texts",
        nargs="*",
        default=[
            "LangFlow makes it easy to orchestrate AI workflows.",
            "Droq nodes can host specialized ML models.",
        ],
        help="Texts to embed. Defaults to two demo sentences.",
    )
    args = parser.parse_args()

    result = {
        "model": MODEL_NAME,
        "inputs": args.texts,
        "embeddings": run(args.texts),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


def get_component_runner() -> tuple[str, str, str, SentenceTransformer]:
    """
    Return metadata + runner for this component.

    Returns:
        (task, model_id, runner_kind, runner_instance)
    """
    model = load_model()
    return ("embeddings", HF_MODEL_ID, "sentence_transformer", model)
