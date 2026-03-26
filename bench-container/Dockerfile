# =============================================================================
# GPUSCALE Benchmark Container — llama.cpp + Meta native inference
# Uses prebuilt llama.cpp image — no compilation needed
# =============================================================================

FROM ghcr.io/ggml-org/llama.cpp:light-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    jq curl \
    && rm -rf /var/lib/apt/lists/*

# uv for fast Python installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project definition and sync dependencies
COPY pyproject.toml /app/pyproject.toml
WORKDIR /app
RUN uv sync --no-dev --no-install-project \
    && uv run python3 -c "from models.llama3.generation import Llama3; print('llama-models OK')"

# Application code
COPY entrypoint.sh /app/entrypoint.sh
COPY scripts/ /app/scripts/
COPY workloads/ /app/workloads/

RUN chmod +x /app/entrypoint.sh /app/scripts/*.sh
RUN mkdir -p /models /results

ENTRYPOINT ["/app/entrypoint.sh"]
