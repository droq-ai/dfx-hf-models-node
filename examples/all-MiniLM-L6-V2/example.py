"""Example usage of the all-MiniLM-L6-V2 local embedding runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT_DIR / "dfx" / "components" / "all-MiniLM-L6-V2" / "run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("all_minilm_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_runner()
    demo_texts = [
        "AgentQL enables structured data extraction from web sources.",
        "Sentence transformers provide strong embeddings out-of-the-box.",
    ]
    embeddings = runner.run(demo_texts)
    print(json.dumps({"inputs": demo_texts, "embedding_shapes": [len(vec) for vec in embeddings]}, indent=2))


if __name__ == "__main__":
    main()

