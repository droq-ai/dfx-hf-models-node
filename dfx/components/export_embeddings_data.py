"""Utility runner that aggregates raw items and embeddings from multiple inputs.

This runner receives data from DFXEmbeddingsComponent outputs and aggregates them.

Expected input formats from DFXEmbeddingsComponent (via NATS):
1. Single text: {"text": "...", "embeddings": [...], "model": "..."}
2. Multiple texts: {"texts": [...], "embeddings": [[...], [...]], "model": "...", "count": N}
3. Nested Data: {"data": {"text": "...", "embeddings": [...], "model": "..."}}

Output format:
{
    "data": [{"text": "...", "model": "..."}, ...],  # Raw items for downstream processing
    "embeddings": [{"id": "...", "vector": [...], "text": "...", "payload": {...}}, ...]  # Vector store format
}
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ExportEmbeddingsDataRunner:
    """Callable runner that merges raw data and embeddings from multiple inputs."""

    def __call__(self, inputs: Any, **parameters: Any) -> dict[str, Any]:
        data_inputs = self._extract_data_inputs(inputs, parameters)
        payloads = list(self._iter_payloads(data_inputs))

        merged_items = self._collect_raw_items(payloads)
        merged_embeddings = self._collect_embeddings(payloads)

        return {
            "data": merged_items,
            "embeddings": merged_embeddings,
        }

    @staticmethod
    def _extract_data_inputs(inputs: Any, parameters: dict[str, Any]) -> list[Any]:
        """Prefer explicit data_inputs in parameters; fall back to request inputs."""
        if "data_inputs" in parameters and parameters["data_inputs"]:
            candidates = parameters["data_inputs"]
        elif inputs:
            candidates = inputs
        else:
            raise ValueError("ExportEmbeddingsDataComponent requires 'data_inputs' to be provided.")

        if isinstance(candidates, list):
            return candidates
        return [candidates]

    @staticmethod
    def _iter_payloads(data_inputs: Iterable[Any]) -> Iterable[dict]:
        """Yield normalized payloads from input data.

        Handles various input structures including:
        - PrecomputedEmbeddings: {"type": "PrecomputedEmbeddings", "vectors": [...], "texts": [...]}
        - Data objects: {"data": {...}}
        - Direct embedding payloads: {"text": "...", "embeddings": [...]}
        """
        for item in data_inputs:
            if isinstance(item, dict):
                # Check for PrecomputedEmbeddings format
                if item.get("type") == "PrecomputedEmbeddings":
                    # This is a merged PrecomputedEmbeddings object
                    yield item
                # Check for nested "data" key (from Data.data serialization)
                elif "data" in item and isinstance(item["data"], dict):
                    inner = item["data"]
                    # Check if inner data is PrecomputedEmbeddings
                    if inner.get("type") == "PrecomputedEmbeddings":
                        yield inner
                    else:
                        yield inner
                else:
                    yield item
            else:
                yield {"value": item}

    @staticmethod
    def _collect_raw_items(payloads: list[dict]) -> list[Any]:
        """Collect raw data items (text + metadata) from payloads."""
        merged: list[Any] = []

        for payload in payloads:
            # Handle PrecomputedEmbeddings format
            if payload.get("type") == "PrecomputedEmbeddings":
                texts = payload.get("texts", [])
                vectors = payload.get("vectors", []) or payload.get("embeddings", [])
                for i, text in enumerate(texts):
                    item = {"text": text}
                    if i < len(vectors):
                        item["embeddings"] = vectors[i]
                    merged.append(item)
                continue

            # Collect from explicit "items" array
            if isinstance(payload.get("items"), list):
                merged.extend(payload["items"])
                continue

            # Collect single text entry
            if payload.get("text"):
                merged.append(
                    {
                        "text": payload.get("text"),
                        "model": payload.get("model"),
                        "embeddings": payload.get("embeddings"),
                    }
                )
                continue

            # Collect multiple texts
            if payload.get("texts") and isinstance(payload["texts"], list):
                texts = payload["texts"]
                embeddings = payload.get("embeddings", []) or payload.get("vectors", [])
                model = payload.get("model")
                for i, text in enumerate(texts):
                    item = {"text": text, "model": model}
                    if i < len(embeddings):
                        item["embeddings"] = embeddings[i]
                    merged.append(item)
                continue

            # Check nested structure (legacy format)
            nested_data = payload.get("locals", {}).get("output", {}).get("data", {})
            if isinstance(nested_data, dict):
                if isinstance(nested_data.get("items"), list):
                    merged.extend(nested_data["items"])
                elif nested_data.get("text"):
                    merged.append(
                        {
                            "text": nested_data.get("text"),
                            "model": nested_data.get("model"),
                            "embeddings": nested_data.get("embeddings"),
                        }
                    )

        return merged

    @staticmethod
    def _collect_embeddings(payloads: list[dict]) -> list[dict]:
        """Collect embeddings in vector-store compatible format.

        Returns list of entries with:
        - id: Deterministic hash of text content
        - vector: Embedding array (for vector DBs)
        - text: Original text
        - payload: Metadata dict (for Qdrant)
        - metadata: Same as payload (for other DBs)
        """
        import hashlib

        merged: list[dict[str, Any]] = []

        def to_entry(
            text: str, vector: list[float], extra_metadata: dict | None = None
        ) -> dict[str, Any]:
            """Create a vector-store compatible entry."""
            # Normalize text
            if isinstance(text, dict):
                text = text.get("text", str(text))
            elif isinstance(text, list):
                # Join list items
                text = " | ".join(
                    (
                        item.get("title", item.get("text", str(item)))
                        if isinstance(item, dict)
                        else str(item)
                    )
                    for item in text
                )
            else:
                text = str(text) if text else ""

            # Generate deterministic ID from text content
            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            entry_id = f"emb-{text_hash}"

            # Build metadata
            metadata = {"text": text}
            if extra_metadata:
                for key, value in extra_metadata.items():
                    if key not in {"embeddings", "vector", "vectors", "type", "texts"}:
                        metadata[key] = value

            return {
                "id": entry_id,
                "vector": vector,  # Standard field name for vector DBs
                "text": text,
                "payload": metadata,  # Qdrant uses "payload"
                "metadata": metadata,  # Other DBs use "metadata"
                "embeddings": vector,  # Backward compatibility
            }

        for payload in payloads:
            # Handle PrecomputedEmbeddings format (from DFXEmbeddingsComponent)
            if payload.get("type") == "PrecomputedEmbeddings":
                vectors = payload.get("vectors", [])
                texts = payload.get("texts", [])

                for i, vector in enumerate(vectors):
                    if not isinstance(vector, list):
                        continue
                    text = texts[i] if i < len(texts) else f"item_{i}"
                    entry = to_entry(text, vector, {"model": payload.get("model")})
                    merged.append(entry)
                continue

            # Get embeddings - check both "embeddings" and "vectors" keys
            embeddings = payload.get("embeddings") or payload.get("vectors")

            if not embeddings or not isinstance(embeddings, list):
                continue

            # Check if it's a single vector (list of floats) or multiple vectors (list of lists)
            if embeddings and isinstance(embeddings[0], int | float):
                # Single vector - pair with single text
                text = payload.get("text", "")
                entry = to_entry(text, embeddings, payload)
                merged.append(entry)

            elif embeddings and isinstance(embeddings[0], list):
                # Multiple vectors - pair with texts array
                texts = payload.get("texts", [])

                for i, vector in enumerate(embeddings):
                    if not isinstance(vector, list):
                        continue
                    # Get corresponding text
                    text = texts[i] if i < len(texts) else f"item_{i}"
                    entry = to_entry(text, vector, {"model": payload.get("model")})
                    merged.append(entry)

            # Also check nested structure (legacy format)
            nested_data = payload.get("locals", {}).get("output", {}).get("data", {})
            if isinstance(nested_data, dict):
                nested_embeddings = nested_data.get("embeddings")
                if isinstance(nested_embeddings, list) and nested_embeddings:
                    if isinstance(nested_embeddings[0], int | float):
                        text = nested_data.get("text", "")
                        entry = to_entry(text, nested_embeddings, nested_data)
                        merged.append(entry)

        return merged


def get_component_runner():
    """Return task metadata and the callable runner."""
    runner = ExportEmbeddingsDataRunner()
    return (
        "export_embeddings_data",
        "ExportEmbeddingsDataComponent",
        "pipeline",
        runner,
    )


class ExportEmbeddingsDataComponent:
    """
    Lightweight placeholder component so Langflow's registrar can import this module.

    The actual execution logic lives in ``ExportEmbeddingsDataRunner`` above; the
    backend only needs a class with a matching name to satisfy component discovery.
    """

    name = "ExportEmbeddingsDataComponent"
    description = (
        "Merge outputs from multiple embedding components into separate data and "
        "embedding streams."
    )


class DFXExportEmbeddingsDataComponent(ExportEmbeddingsDataComponent):
    """
    Registry-facing alias that matches the component name advertised by the node.

    DroqFlow looks up `DFXExportEmbeddingsDataComponent` in the registry, so expose
    this alias to ensure component discovery and routing succeed without having to
    duplicate implementation details.
    """

    name = "DFXExportEmbeddingsDataComponent"


__all__ = ["get_component_runner"]
