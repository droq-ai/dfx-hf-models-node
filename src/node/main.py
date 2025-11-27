#!/usr/bin/env python3
"""
DFX Hugging Face Models Node.

Runs a FastAPI service that exposes locally hosted Hugging Face models (embeddings, text, etc.)
and integrates with Droq's NATS-based workflow orchestration system.
"""

import logging
import os

import uvicorn

try:
    from .logger import setup_logging
except ImportError:  # pragma: no cover - optional dependency
    setup_logging = None

from .api import app

logger = logging.getLogger(__name__)


def main() -> None:
    """Configure logging and launch the FastAPI server."""
    if setup_logging:
        setup_logging()
    else:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8006"))
    reload = os.getenv("RELOAD", "true").lower() == "true"

    logger.info("Starting DFX HF Models Node on %s:%d (reload=%s)", host, port, reload)

    if reload:
        uvicorn.run(
            "node.api:app",
            host=host,
            port=port,
            reload=True,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
