"""API tests for the DFX HF Models Node."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from node.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"


class _FakeVector:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeSentenceTransformer:
    def encode(self, texts, convert_to_numpy=True):
        return [_FakeVector([float(len(text)), 2.0, 3.0]) for text in texts]


class _FakePipeline:
    def __call__(self, inputs, **kwargs):
        return {"echo": inputs, "params": kwargs}


@patch(
    "node.api._load_component_runner",
    return_value=("embeddings", "fake-model", "sentence_transformer", _FakeSentenceTransformer()),
)
def test_execute_embeddings(mock_loader):
    response = client.post(
        "/api/v1/execute",
        json={"component": "MiniLMEmbeddingsComponent", "inputs": ["hello", "world"]},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["component"] == "MiniLMEmbeddingsComponent"
    assert data["model_id"] == "fake-model"
    assert data["task"] == "embeddings"
    assert len(data["output"]) == 2

    mock_loader.assert_called_once_with("MiniLMEmbeddingsComponent")


@patch(
    "node.api._load_component_runner",
    return_value=("text-generation", "fake-model", "pipeline", _FakePipeline()),
)
def test_execute_pipeline(mock_loader):
    payload = {
        "component": "SomePipelineComponent",
        "inputs": "hello world",
        "parameters": {"max_new_tokens": 5},
    }
    response = client.post("/api/v1/execute", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["component"] == "SomePipelineComponent"
    assert data["task"] == "text-generation"
    assert data["output"]["echo"] == "hello world"
    assert data["output"]["params"]["max_new_tokens"] == 5

    mock_loader.assert_called_once_with("SomePipelineComponent")
