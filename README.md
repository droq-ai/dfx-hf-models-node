# DFX Hugging Face Models Node

A Droqflow node providing execution of Hugging Face models.

## Quick Start

```bash
git clone git@github.com:droq-ai/dfx-hf-models-node.git
cd dfx-hf-models-node
uv sync

# Configure node.json with your node information
# Configure environment variables

# Test locally
PYTHONPATH=src uv run python -m node.main

# Run with Docker
docker compose up
```

## Docker

```bash
# Build
docker build -t your-node:latest .

# Run
docker run -p 8006:8006 \
  -e NODE_NAME=dfx-hf-models-node \
  -e NATS_URL=nats://localhost:4222 \
  droq-ai/dfx-hf-models-node:latest
```

## Development

```bash
# Run tests
PYTHONPATH=src uv run pytest

# Format code
uv run black src/ tests/
uv run ruff check --fix src/ tests/

# Add dependencies
uv add package-name
```



## Documentation

- [Usage Guide](docs/usage.md)
- [Configuration Guide](docs/node-configuration.md)
- [NATS Examples](docs/nats.md)

## License

Apache License 2.0