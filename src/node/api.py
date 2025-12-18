"""
FastAPI application for the DFX Hugging Face Models Node.

Provides a generic /api/v1/execute endpoint to run locally hosted Hugging Face models
and optionally publish their results to Droq's NATS streams.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .nats import NATSClient

logger = logging.getLogger(__name__)
app = FastAPI(title="DFX Hugging Face Models Node", version="0.1.0")

NODE_ROOT = Path(__file__).resolve().parents[2]
NODE_CONFIG_PATH = NODE_ROOT / "node.json"
_runner_cache: dict[str, dict[str, Any]] = {}
_nats_client: NATSClient | None = None


def _load_supported_components() -> dict[str, str]:
    try:
        with NODE_CONFIG_PATH.open("r") as file:
            config = json.load(file)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load node configuration at %s: %s", NODE_CONFIG_PATH, exc)
        return {}

    components = config.get("components", {})
    mapping: dict[str, str] = {}
    for name, metadata in components.items():
        module_path = metadata.get("path")
        if not module_path:
            logger.warning("Component '%s' missing 'path' in node.json; skipping", name)
            continue
        mapping[name] = module_path
    logger.info("Registered %d component(s) from node configuration", len(mapping))
    return mapping


COMPONENT_MODULES = _load_supported_components()


class ComponentState(BaseModel):
    """Component state for langflow executor API format."""

    component_class: str
    component_module: str
    component_code: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_values: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    display_name: str | None = None
    component_id: str | None = None
    stream_topic: str | None = None
    attributes: dict[str, Any] | None = None


class ExecutionRequest(BaseModel):
    """Generic execution payload - supports both dfx and langflow executor formats."""

    # DFX format fields
    component: str | None = Field(None, description="Registered component name (DFX format).")
    inputs: Any = Field(None, description="Task-specific inputs (DFX format).")
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="Optional parameters (DFX format).",
    )
    publish_subject: str | None = Field(
        default=None,
        description="Optional NATS subject to publish the result to (DFX format).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Extra metadata to include when publishing (DFX format).",
    )

    # Langflow executor format fields
    component_state: ComponentState | None = Field(
        None, description="Component state (Langflow executor format)."
    )
    method_name: str | None = Field(
        None, description="Method name to execute (Langflow executor format)."
    )
    is_async: bool = Field(False, description="Whether method is async (Langflow executor format).")
    timeout: int = Field(30, description="Execution timeout (Langflow executor format).")
    message_id: str | None = Field(
        None, description="Message ID for tracking (Langflow executor format)."
    )


class ExecutionResponse(BaseModel):
    """Result returned by an execution request."""

    message_id: str
    component: str
    task: str
    model_id: str
    output: Any
    execution_time: float
    published_subject: str | None = None


def _resolve_models_root() -> Path:
    """Determine the root directory that contains cached Hugging Face models."""
    env_path = os.getenv("HF_MODELS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    # Default to repository models directory
    return Path(__file__).resolve().parents[2] / "models"


_RunnerKind = Literal["sentence_transformer", "pipeline"]


def _load_component_runner(component_name: str) -> tuple[str, str, _RunnerKind, Any]:
    """Load (or fetch from cache) the runner for a registered component."""
    module_path = COMPONENT_MODULES.get(component_name)
    if not module_path:
        raise HTTPException(
            status_code=400,
            detail=f"Component '{component_name}' is not supported by this node.",
        )

    cached = _runner_cache.get(component_name)
    if cached:
        return cached["task"], cached["model_id"], cached["kind"], cached["runner"]

    try:
        module = import_module(module_path)
        loader = getattr(module, "get_component_runner")
    except Exception as exc:  # noqa: BLE001
        msg = f"Component module '{module_path}' must define get_component_runner(): {exc}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg) from exc

    try:
        task, model_id, runner_kind, runner = loader()
    except Exception as exc:  # noqa: BLE001
        logger.error("Component loader for '%s' failed: %s", component_name, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _runner_cache[component_name] = {
        "task": task,
        "model_id": model_id,
        "kind": runner_kind,
        "runner": runner,
    }
    return task, model_id, runner_kind, runner


async def _get_nats_client() -> NATSClient | None:
    """Initialize (if necessary) and return a shared NATS client."""
    global _nats_client

    if _nats_client is not None:
        return _nats_client

    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    stream_name = os.getenv("STREAM_NAME", "droq-stream")

    try:
        client = NATSClient(nats_url=nats_url, stream_name=stream_name)
        await client.connect()
        logger.info("Connected to NATS at %s (stream=%s)", nats_url, stream_name)
        _nats_client = client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to connect to NATS (%s). Running in degraded mode.", exc)
        _nats_client = None

    return _nats_client


@app.on_event("shutdown")
async def _shutdown_event() -> None:
    """Release NATS resources when the server stops."""
    global _nats_client
    if _nats_client:
        await _nats_client.close()
        _nats_client = None


def _normalize_embedding_inputs(inputs: Any) -> list[str]:
    if isinstance(inputs, str):
        return [inputs]
    if isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
        return inputs
    raise HTTPException(
        status_code=400, detail="Embedding tasks require a string or list of strings as inputs."
    )


@app.post("/api/v1/execute")
async def execute(request: ExecutionRequest) -> dict[str, Any]:
    """Execute a Hugging Face task (embeddings, generation, etc.) locally and optionally publish to NATS.

    Supports both DFX format and Langflow executor format.
    """
    # Detect which format we're receiving
    if request.component_state is not None:
        # Langflow executor format
        return await _execute_langflow_format(request)
    elif request.component is not None:
        # DFX format
        return await _execute_dfx_format(request)
    else:
        raise HTTPException(
            status_code=422,
            detail="Request must include either 'component' (DFX format) or 'component_state' (Langflow executor format).",
        )


async def _execute_dfx_format(request: ExecutionRequest) -> dict[str, Any]:
    """Execute using DFX format."""
    task, model_id, runner_kind, runner = _load_component_runner(request.component)

    start_time = time.perf_counter()
    if runner_kind == "sentence_transformer":
        texts = _normalize_embedding_inputs(request.inputs)
        try:
            embeddings = await asyncio.to_thread(
                runner.encode,
                texts,
                convert_to_numpy=True,
            )
        except Exception as exc:  # pragma: no cover
            msg = f"Failed to compute embeddings: {exc}"
            logger.error(msg, exc_info=True)
            raise HTTPException(status_code=500, detail=msg) from exc
        output: Any = [vector.tolist() for vector in embeddings]
    else:
        parameters = request.parameters or {}
        try:
            output = await asyncio.to_thread(runner, request.inputs, **parameters)
        except Exception as exc:  # pragma: no cover
            msg = f"Pipeline execution failed: {exc}"
            logger.error(msg, exc_info=True)
            raise HTTPException(status_code=500, detail=msg) from exc

    execution_time = time.perf_counter() - start_time
    message_id = str(uuid.uuid4())

    publish_subject = request.publish_subject or os.getenv("HF_EXECUTE_SUBJECT")
    if publish_subject:
        payload = {
            "message_id": message_id,
            "component": request.component,
            "task": task,
            "model_id": model_id,
            "output": output,
            "inputs": request.inputs,
            "parameters": request.parameters or {},
            "metadata": request.metadata or {},
        }
        logger.info(
            f"[NATS] [DFX Format] Attempting to publish to subject: {publish_subject} with message_id: {message_id}"
        )
        logger.info(f"[NATS] [DFX Format] Publish data keys: {list(payload.keys())}")
        logger.info(
            f"[NATS] [DFX Format] Component: {request.component}, task: {task}, model_id: {model_id}"
        )
        logger.info(
            f"[NATS] [DFX Format] Output type: {type(output).__name__}, output length: {len(output) if isinstance(output, list) else 'N/A'}"
        )
        # Log full payload (truncated for very large outputs)
        payload_str = json.dumps(payload, indent=2, default=str)
        if len(payload_str) > 2000:
            logger.info(
                f"[NATS] [DFX Format] Full publish data (truncated): {payload_str[:2000]}..."
            )
            print(
                f"[NATS] [DFX Format] Full publish data (truncated): {payload_str[:2000]}...",
                flush=True,
            )
        else:
            logger.info(f"[NATS] [DFX Format] Full publish data: {payload_str}")
            print(f"[NATS] [DFX Format] Full publish data: {payload_str}", flush=True)
        nats_client = await _get_nats_client()
        if nats_client:
            try:
                await nats_client.publish(publish_subject, payload)
                logger.info(
                    "[NATS] [DFX Format] ✅ Published execution result (message_id=%s) to subject %s",
                    message_id,
                    publish_subject,
                )
                print(
                    f"[NATS] [DFX Format] ✅ Published execution result (message_id={message_id}) to subject {publish_subject}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[NATS] [DFX Format] ❌ Failed to publish to NATS subject %s: %s",
                    publish_subject,
                    exc,
                )
                print(
                    f"[NATS] [DFX Format] ❌ Failed to publish to NATS subject {publish_subject}: {exc}",
                    flush=True,
                )
        else:
            logger.warning(
                "[NATS] [DFX Format] ⚠️  NATS client unavailable; skipping publish to %s",
                publish_subject,
            )
            print(
                f"[NATS] [DFX Format] ⚠️  NATS client unavailable; skipping publish to {publish_subject}",
                flush=True,
            )
    else:
        publish_subject = None
        logger.debug("[NATS] [DFX Format] No publish_subject provided, skipping NATS publish")

    return ExecutionResponse(
        message_id=message_id,
        component=request.component,
        task=task,
        model_id=model_id,
        output=output,
        published_subject=publish_subject,
        execution_time=execution_time,
    ).model_dump()


async def _execute_langflow_format(request: ExecutionRequest) -> dict[str, Any]:
    """Execute using Langflow executor format."""
    component_state = request.component_state
    component_class = component_state.component_class
    method_name = request.method_name or "generate_embeddings"
    message_id = request.message_id or str(uuid.uuid4())
    stream_topic = component_state.stream_topic

    # Load the component runner
    task, model_id, runner_kind, runner = _load_component_runner(component_class)

    # Handle pipeline components (e.g., export_embeddings_data)
    # They process data_inputs directly, not text for embedding
    if runner_kind == "pipeline":
        parameters = component_state.parameters or {}
        input_values = component_state.input_values or {}
        data_inputs = parameters.get("data_inputs") or input_values.get("data_inputs")

        if not data_inputs:
            raise HTTPException(
                status_code=400, detail="No data_inputs found for pipeline component"
            )

        start_time = time.perf_counter()
        try:
            output = await asyncio.to_thread(runner, data_inputs, **parameters)
        except Exception as exc:
            logger.error(f"Pipeline execution failed: {exc}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Pipeline execution failed: {exc}"
            ) from exc

        execution_time = time.perf_counter() - start_time
        result_data = output if isinstance(output, dict) else {"result": output}

        # Publish to NATS if stream topic provided
        if stream_topic:
            try:
                nats_client = await _get_nats_client()
                if nats_client:
                    publish_data = {
                        "message_id": message_id,
                        "component_id": component_state.component_id,
                        "component_class": component_class,
                        "result": result_data,
                        "result_type": "Data",
                        "execution_time": execution_time,
                    }
                    await nats_client.publish(stream_topic, publish_data)
                    logger.info(f"[NATS] Published pipeline result to {stream_topic}")
            except Exception as e:
                logger.warning(f"[NATS] Failed to publish pipeline result: {e}")

        return {
            "success": True,
            "result": result_data,
            "result_type": "Data",
            "execution_time": execution_time,
        }

    # Extract text inputs from component state
    # For embeddings, inputs can be in input_values or parameters
    texts = []
    if component_state.input_values:
        input_data = component_state.input_values.get("input_data")
        if input_data:
            # Extract texts from various input formats
            if isinstance(input_data, str):
                texts = [input_data]
            elif isinstance(input_data, list):
                texts = [str(item) for item in input_data]
            elif isinstance(input_data, dict):
                # Try to extract text from common keys
                if "text" in input_data:
                    text_val = input_data["text"]
                    texts = (
                        [text_val]
                        if isinstance(text_val, str)
                        else text_val if isinstance(text_val, list) else [str(text_val)]
                    )
                elif "data" in input_data:
                    data_val = input_data["data"]
                    if isinstance(data_val, dict) and "text" in data_val:
                        texts = (
                            [data_val["text"]]
                            if isinstance(data_val["text"], str)
                            else data_val["text"]
                        )
                    else:
                        texts = [str(data_val)]
                else:
                    texts = [str(input_data)]
            else:
                texts = [str(input_data)]

    if not texts:
        raise HTTPException(status_code=400, detail="No text inputs found in component state")

    # Normalize texts
    texts = _normalize_embedding_inputs(texts)

    # Execute embeddings
    start_time = time.perf_counter()
    if runner_kind == "sentence_transformer":
        try:
            embeddings = await asyncio.to_thread(
                runner.encode,
                texts,
                convert_to_numpy=True,
            )
        except Exception as exc:
            msg = f"Failed to compute embeddings: {exc}"
            logger.error(msg, exc_info=True)
            raise HTTPException(status_code=500, detail=msg) from exc
        output = [vector.tolist() for vector in embeddings]
    else:
        parameters = component_state.parameters or {}
        try:
            output = await asyncio.to_thread(runner, texts, **parameters)
        except Exception as exc:
            msg = f"Pipeline execution failed: {exc}"
            logger.error(msg, exc_info=True)
            raise HTTPException(status_code=500, detail=msg) from exc

    execution_time = time.perf_counter() - start_time

    # Check method_name to determine output format
    # - get_embeddings_only: return only the embeddings array
    # - generate_embeddings (default): return full Data object
    if method_name == "get_embeddings_only":
        # Return only embeddings array
        if len(texts) == 1:
            result_data = output[0] if output else []
        else:
            result_data = output
        result_type = "Embeddings"
    else:
        # Return full Data object with text, embeddings, model
        if len(texts) == 1:
            data_content = {
                "text": texts[0],
                "embeddings": output[0] if output else [],
                "model": component_class,
            }
        else:
            data_content = {
                "texts": texts,
                "embeddings": output,
                "model": component_class,
                "count": len(texts),
            }

        # Wrap in Data object structure
        result_data = {
            "data": data_content,
            "text_key": "text" if len(texts) == 1 else None,
            "default_value": "",
        }
        result_type = "Data"

    # Serialize result for NATS publishing
    serialized_result = result_data

    # Publish to NATS
    if stream_topic:
        logger.info(
            f"[NATS] Attempting to publish to topic: {stream_topic} with message_id: {message_id}"
        )
        try:
            nats_client = await _get_nats_client()
            if nats_client:
                # Publish result to NATS with message ID from backend
                publish_data = {
                    "message_id": message_id,
                    "component_id": component_state.component_id,
                    "component_class": component_class,
                    "result": serialized_result,
                    "result_type": result_type,
                    "execution_time": execution_time,
                }
                logger.info(
                    f"[NATS] Publishing to topic: {stream_topic}, message_id: {message_id}, data keys: {list(publish_data.keys())}"
                )
                logger.info(
                    f"[NATS] Publish data preview: component_class={component_class}, result_type={publish_data['result_type']}, execution_time={execution_time:.3f}s"
                )
                logger.info(
                    f"[NATS] Result data keys: {list(serialized_result.keys()) if isinstance(serialized_result, dict) else 'N/A'}"
                )
                # Log full publish data (truncated for very large results)
                publish_data_str = json.dumps(publish_data, indent=2, default=str)
                if len(publish_data_str) > 2000:
                    logger.info(
                        f"[NATS] Full publish data (truncated): {publish_data_str[:2000]}..."
                    )
                    print(
                        f"[NATS] Full publish data (truncated): {publish_data_str[:2000]}...",
                        flush=True,
                    )
                else:
                    logger.info(f"[NATS] Full publish data: {publish_data_str}")
                    print(f"[NATS] Full publish data: {publish_data_str}", flush=True)
                print(
                    f"[NATS] Publishing to topic: {stream_topic}, message_id: {message_id}, data keys: {list(publish_data.keys())}",
                    flush=True,
                )
                # Use the topic directly (already in format: droq.local.public.userid.workflowid.component.out)
                await nats_client.publish(stream_topic, publish_data)
                logger.info(
                    f"[NATS] ✅ Successfully published result to NATS topic: {stream_topic} with message_id: {message_id}"
                )
                print(
                    f"[NATS] ✅ Successfully published result to NATS topic: {stream_topic} with message_id: {message_id}",
                    flush=True,
                )
            else:
                error_msg = f"[NATS] ❌ NATS client unavailable; cannot publish to required topic: {stream_topic}"
                logger.error(error_msg)
                print(error_msg, flush=True)
                raise HTTPException(status_code=500, detail=error_msg)
        except HTTPException:
            raise
        except Exception as exc:
            # NATS publishing is mandatory - fail if it doesn't work
            error_msg = f"Failed to publish to NATS topic {stream_topic}: {exc}"
            logger.error(error_msg, exc_info=True)
            print(f"[NATS] ❌ {error_msg}", flush=True)
            raise HTTPException(status_code=500, detail=error_msg) from exc
    else:
        logger.warning(
            f"[NATS] ⚠️  No stream_topic provided in request, skipping NATS publish. Component: {component_class}"
        )
        print(
            f"[NATS] ⚠️  No stream_topic provided in request, skipping NATS publish. Component: {component_class}",
            flush=True,
        )

    # Return langflow executor format response
    return {
        "result": serialized_result,
        "success": True,
        "result_type": result_type,
        "execution_time": execution_time,
        "message_id": message_id,
        "updated_attributes": None,
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "healthy", "service": "dfx-hf-models-node"}


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint that describes available functionality."""
    return {
        "service": "DFX HF Models Node",
        "version": "0.1.0",
        "model_directory": str(_resolve_models_root()),
        "endpoints": {
            "execute": "/api/v1/execute",
            "health": "/health",
        },
    }
