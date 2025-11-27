"""Langflow component for all-MiniLM-L6-V2 embeddings model."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

from lfx.base.embeddings.model import LCEmbeddingsModel
from lfx.field_typing import Embeddings
from lfx.io import IntInput

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
except ImportError:
    SentenceTransformer = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
MODEL_NAME = "all-MiniLM-L6-V2"
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-V2"
DOWNLOAD_SCRIPT = ROOT_DIR / "scripts" / "download.sh"


class MiniLMEmbeddingsComponent(Embeddings):
    """Droq Embeddings wrapper for local execution of SentenceTransformer model."""

    def __init__(self, model_path: Path) -> None:
        """Initialize with path to local model."""
        if SentenceTransformer is None:
            msg = "sentence-transformers must be installed. Install it with: uv pip install sentence-transformers"
            raise ImportError(msg)
        self.model = SentenceTransformer(str(model_path))
        self.model_path = model_path

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        if not text:
            return []
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class MiniLMEmbeddingsComponent(LCEmbeddingsModel):
    """DroqFlow component for all-MiniLM-L6-V2 embeddings using local model."""

    display_name = "MiniLM Embeddings"
    description = "Generate embeddings using the locally stored all-MiniLM-L6-V2 SentenceTransformer model."
    documentation: str = "https://www.sbert.net/docs/pretrained_models.html"
    icon = "binary"
    name = "MiniLMEmbeddingsComponent"
    category = "models"

    inputs = [
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            info="Number of texts to embed in a single batch.",
            advanced=True,
            value=32,
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        """Build and return the local MiniLM embeddings model."""
        model_path = ensure_model()
        return MiniLMEmbeddingsComponent(model_path)


def ensure_model() -> Path:
    """Ensure the MiniLM model assets exist locally, downloading them if necessary."""
    target_dir = MODELS_DIR / MODEL_NAME
    if target_dir.exists():
        return target_dir.resolve()

    if not DOWNLOAD_SCRIPT.exists():
        raise FileNotFoundError(f"Download script not found at {DOWNLOAD_SCRIPT}")

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(DOWNLOAD_SCRIPT), HF_MODEL_ID, str(target_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Failed to download {HF_MODEL_ID}: {detail}")

    if not target_dir.exists():
        raise RuntimeError(f"Download reported success but assets missing at {target_dir}")

    return target_dir.resolve()


def get_component_runner() -> Tuple[str, str, str, MiniLMEmbeddingsComponent]:
    """
    Return task, model identifier, runner kind, and runner instance for this component.
    """
    model_path = ensure_model()
    if SentenceTransformer is None:
        msg = "sentence-transformers must be installed. Install it with: uv pip install sentence-transformers"
        raise ImportError(msg)
    runner = SentenceTransformer(str(model_path))
    return ("embeddings", HF_MODEL_ID, "sentence_transformer", runner)