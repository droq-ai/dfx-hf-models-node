"""LLMLingua - Reduce token count while preserving content quality.

Uses LLMLingua-2 (Microsoft's prompt compression) to intelligently reduce
tokens by ~10% without losing important content.

Expected input:
- {"text": "...", "rate": 0.9}  # rate=0.9 means keep 90% of tokens
- Or via parameters: inputs={"text": "..."}, rate=0.9

Output:
{
    "compressed_text": "...",
    "original_chars": N,
    "compressed_chars": M,
    "reduction_percent": float
}
"""

from __future__ import annotations

from typing import Any

# Lazy import to avoid loading model at module import time
_compressor = None


def _clean_json_noise(text: str) -> str:
    """Repair and clean JSON/text using jsonrepair."""
    from json_repair import repair_json

    try:
        # Try to repair if it looks like JSON
        if text.strip().startswith(("{", "[")):
            repaired = repair_json(text)
            return str(repaired).strip()
    except Exception:
        raise ValueError("Failed to repair JSON/text using jsonrepair.")

    return text.strip()


def _get_compressor():
    """Lazy load the LLMLingua-2 compressor."""
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor

        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _compressor


def _compress_text(text: str, rate: float = 0.9) -> dict[str, Any]:
    """
    Compress text using LLMLingua-2.

    Args:
        text: Input text to compress
        rate: Fraction of tokens to keep (0.9 = keep 90%, reduce by 10%)

    Returns:
        Dictionary with compressed_text and stats
    """
    compressor = _get_compressor()

    original_chars = len(text)

    # Process in chunks if text is large (LLMLingua-2 has 512 token limit)
    chunk_size = 500  # chars, safe for 512 token limit

    if original_chars <= chunk_size:
        # Small text - process directly
        result = compressor.compress_prompt(
            text,
            rate=rate,
            force_tokens=["\n", "?", ":", "-", "*", "#", "%"],
        )
        compressed_text = result["compressed_prompt"]
    else:
        # Large text - process in chunks
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        compressed_chunks = []

        for chunk in chunks:
            result = compressor.compress_prompt(
                chunk,
                rate=rate,
                force_tokens=["\n", "?", ":", "-", "*", "#", "%"],
            )
            compressed_chunks.append(result["compressed_prompt"])

        compressed_text = "".join(compressed_chunks)

    compressed_chars = len(compressed_text)
    reduction = (1 - compressed_chars / original_chars) * 100 if original_chars > 0 else 0

    # Return in Data-compatible format so _deserialize_result can properly reconstruct it
    # Data objects expect either "data" key or "text_key" key
    # Only return text (compressed text) - stats are for status display only
    return {
        "data": {
            "text": compressed_text,  # The compressed text (main output)
            "original_chars": original_chars,  # Stats for status display
            "compressed_chars": compressed_chars,
            "reduction_percent": round(reduction, 1),
        },
        "text_key": "text",  # Also include text_key for Data compatibility
    }


class LLMLinguaRunner:
    """Callable runner that compresses text to reduce token count."""

    def __call__(self, inputs: Any, **parameters: Any) -> dict[str, Any]:
        # Extract parameters
        text = self._extract_text(inputs, parameters)
        rate = parameters.get("rate", 0.9)  # Default: keep 90% (10% reduction)

        # Clean JSON noise before compression
        text = _clean_json_noise(text)

        # Compress
        return _compress_text(text, rate=rate)

    @staticmethod
    def _extract_text(inputs: Any, parameters: dict[str, Any]) -> str:
        """Extract text from inputs or parameters, ignoring metadata.
        
        Supports:
        - Message objects (with .text or .data.text attributes)
        - Dict structures with data.text or text fields
        - Plain strings
        - Lists of the above
        """
        # Check parameters first
        if "text" in parameters and parameters["text"]:
            text = parameters["text"]
            # Handle Message object
            if hasattr(text, "text"):
                return str(text.text)
            if hasattr(text, "data") and hasattr(text.data, "text"):
                return str(text.data.text)
            # If it's a dict, check for data.items array first
            if isinstance(text, dict):
                data = text.get("data", {})
                if isinstance(data, dict) and "items" in data and isinstance(data["items"], list) and data["items"]:
                    # Extract text from all items and merge
                    texts = []
                    for item in data["items"]:
                        if isinstance(item, dict):
                            item_text = None
                            if "data" in item and isinstance(item["data"], dict):
                                item_text = item["data"].get("text")
                            if not item_text and "text" in item:
                                item_text = item["text"]
                            if item_text:
                                texts.append(str(item_text))
                    if texts:
                        return "\n".join(texts)
                # Fallback to single text field
                return str(text.get("text") or data.get("text", ""))
            return str(text)

        if "input_text" in parameters and parameters["input_text"]:
            text = parameters["input_text"]
            # Handle Message object
            if hasattr(text, "text"):
                return str(text.text)
            if hasattr(text, "data") and hasattr(text.data, "text"):
                return str(text.data.text)
            # If it's a dict, check for data.items array first
            if isinstance(text, dict):
                data = text.get("data", {})
                if isinstance(data, dict) and "items" in data and isinstance(data["items"], list) and data["items"]:
                    # Extract text from all items and merge
                    texts = []
                    for item in data["items"]:
                        if isinstance(item, dict):
                            item_text = None
                            if "data" in item and isinstance(item["data"], dict):
                                item_text = item["data"].get("text")
                            if not item_text and "text" in item:
                                item_text = item["text"]
                            if item_text:
                                texts.append(str(item_text))
                    if texts:
                        return "\n".join(texts)
                # Fallback to single text field
                return str(text.get("text") or data.get("text", ""))
            return str(text)

        # Check inputs
        if isinstance(inputs, str):
            return inputs

        # Handle Message object
        if hasattr(inputs, "text"):
            return str(inputs.text)
        if hasattr(inputs, "data") and hasattr(inputs.data, "text"):
            return str(inputs.data.text)

        if isinstance(inputs, dict):
            # Priority 1: Check data.items (array of Data objects - merge all texts)
            if "data" in inputs and isinstance(inputs["data"], dict):
                data = inputs["data"]
                # Check for items array (multiple Data objects)
                if "items" in data and isinstance(data["items"], list) and data["items"]:
                    # Extract text from all items and merge
                    texts = []
                    for item in data["items"]:
                        if isinstance(item, dict):
                            # Try to get text from item.data.text or item.text
                            item_text = None
                            if "data" in item and isinstance(item["data"], dict):
                                item_text = item["data"].get("text")
                            if not item_text and "text" in item:
                                item_text = item["text"]
                            if item_text:
                                texts.append(str(item_text))
                    if texts:
                        # Merge all texts with newlines
                        return "\n".join(texts)
                
                # Check data.text (single text field)
                text = data.get("text")
                if text:
                    return str(text)
            
            # Priority 2: Check top-level text field
            if "text" in inputs:
                text = inputs["text"]
                # Handle Message object
                if hasattr(text, "text"):
                    return str(text.text)
                if hasattr(text, "data") and hasattr(text.data, "text"):
                    return str(text.data.text)
                # If text itself is a dict, extract nested text
                if isinstance(text, dict):
                    return str(text.get("text") or text.get("data", {}).get("text", ""))
                return str(text)

        if isinstance(inputs, list) and inputs:
            # If list contains multiple items, merge all texts
            texts = []
            for item in inputs:
                if isinstance(item, str):
                    texts.append(item)
                elif hasattr(item, "text"):
                    texts.append(str(item.text))
                elif hasattr(item, "data") and hasattr(item.data, "text"):
                    texts.append(str(item.data.text))
                elif isinstance(item, dict):
                    # Check for data.items array first
                    data = item.get("data", {})
                    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list) and data["items"]:
                        # Extract text from all items in this item's items array
                        for sub_item in data["items"]:
                            if isinstance(sub_item, dict):
                                sub_item_text = None
                                if "data" in sub_item and isinstance(sub_item["data"], dict):
                                    sub_item_text = sub_item["data"].get("text")
                                if not sub_item_text and "text" in sub_item:
                                    sub_item_text = sub_item["text"]
                                if sub_item_text:
                                    texts.append(str(sub_item_text))
                    else:
                        # Check data.text
                        if "data" in item and isinstance(item["data"], dict):
                            text = item["data"].get("text")
                            if text:
                                texts.append(str(text))
                        # Then check top-level text
                        elif "text" in item:
                            texts.append(str(item["text"]))
            
            if texts:
                return "\n".join(texts)
            
            # Fallback: just use first item
            first = inputs[0]
            if isinstance(first, str):
                return first
            if hasattr(first, "text"):
                return str(first.text)
            if hasattr(first, "data") and hasattr(first.data, "text"):
                return str(first.data.text)
            if isinstance(first, dict):
                # Check data.text first
                if "data" in first and isinstance(first["data"], dict):
                    text = first["data"].get("text")
                    if text:
                        return str(text)
                # Then check top-level text
                return str(first.get("text", ""))

        raise ValueError("LLMLinguaComponent requires 'text' or 'input_text' to be provided.")


def get_component_runner():
    """Return task metadata and the callable runner."""
    runner = LLMLinguaRunner()
    return (
        "llmlingua",
        "LLMLinguaComponent",
        "pipeline",
        runner,
    )


class LLMLinguaComponent:
    """
    Lightweight placeholder component for Langflow's registrar.

    The actual execution logic lives in LLMLinguaRunner above.
    """

    name = "LLMLinguaComponent"
    display_name = "LLMLingua"
    description = (
        "Reduce token count by ~10% while preserving content quality using "
        "LLMLingua-2 (Microsoft's prompt compression). Intelligently removes "
        "redundant tokens without losing important information."
    )
    icon = "compress"
    category = "processing"


__all__ = ["get_component_runner", "LLMLinguaComponent"]
