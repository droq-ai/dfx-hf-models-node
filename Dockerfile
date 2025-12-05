FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files needed for installation
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/
COPY dfx/ ./dfx/

# Install dependencies and the package
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --system -r requirements.txt && \
    uv pip install --system .

# Copy node configuration, models, and startup script
COPY node.json ./
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY start-local.sh ./

# Create non-root user and make script executable
RUN useradd -m -u 1000 nodeuser && \
    chown -R nodeuser:nodeuser /app && \
    chmod +x /app/start-local.sh
USER nodeuser

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV NODE_PORT=8000

EXPOSE ${NODE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["./start-local.sh"]

