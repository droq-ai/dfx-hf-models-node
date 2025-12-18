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

    return {
        "compressed_text": compressed_text,
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "reduction_percent": round(reduction, 1),
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
        """Extract text from inputs or parameters."""
        # Check parameters first
        if "text" in parameters and parameters["text"]:
            return str(parameters["text"])

        if "input_text" in parameters and parameters["input_text"]:
            return str(parameters["input_text"])

        # Check inputs
        if isinstance(inputs, str):
            return inputs

        if isinstance(inputs, dict):
            if "text" in inputs:
                return str(inputs["text"])
            if "data" in inputs and isinstance(inputs["data"], dict):
                return str(inputs["data"].get("text", ""))

        if isinstance(inputs, list) and inputs:
            first = inputs[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
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
