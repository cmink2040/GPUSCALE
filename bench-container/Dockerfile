# =============================================================================
# GPUSCALE Benchmark Container — llama.cpp + Meta native inference
# Uses prebuilt llama.cpp image — no compilation needed
#
# Build variants:
#   CUDA 12.4 (default): supports P100, V100, 3090, 4090, A100, H100
#   CUDA 13.1:           supports 3090+, 4090, 5090, A100, H100, B200
# =============================================================================

ARG BASE_IMAGE=ghcr.io/ggml-org/llama.cpp:light-cuda
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    jq curl \
    && rm -rf /var/lib/apt/lists/*

# uv for fast Python installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project definition and sync dependencies
COPY pyproject.toml /app/pyproject.toml
WORKDIR /app
RUN uv sync --python 3.14 --no-dev --no-install-project \
    && uv run python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}'); from models.llama3.generation import Llama3; print('llama-models OK')"

# Application code
COPY entrypoint.sh /app/entrypoint.sh
COPY scripts/ /app/scripts/
COPY workloads/ /app/workloads/

RUN chmod +x /app/entrypoint.sh /app/scripts/*.sh
RUN mkdir -p /models /results

ENTRYPOINT ["/app/entrypoint.sh"]
