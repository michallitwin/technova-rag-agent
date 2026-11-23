FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# NEW: keep the embedding model inside /app so appuser can read it
ENV HF_HOME=/app/.cache/huggingface

# Build the orders database and the vector index inside the image
# (the embedding model is downloaded once, at build time)
RUN uv run --no-sync python -m scripts.create_db \
 && uv run --no-sync python -m src.rag.pipeline

# At runtime use the model baked into the image, no Hugging Face calls
ENV HF_HUB_OFFLINE=1

# Run as a normal user, not root (least privilege)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]