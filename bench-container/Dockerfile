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

# Upstream llama.cpp :light-cuda* installs llama-cli to /app and makes it the
# image ENTRYPOINT — /app is NOT on $PATH by default. We override ENTRYPOINT
# to our harness, so we must ensure llama-cli resolves by name. Put /app on
# PATH explicitly; run_llama_cpp.sh also has its own dynamic discovery as a
# second line of defense. Together these make the bench container agnostic
# to future base-image layout shuffles.
ENV PATH=/app:/usr/local/bin:$PATH

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

# Build-time smoke test: fail the image build loud and early if the wrapper
# scripts' hard requirements aren't satisfiable. This catches base-image swaps
# that move binaries out of PATH before the broken image ships.
RUN set -e; \
    command -v llama-cli >/dev/null 2>&1 || { \
        echo "BUILD SMOKE TEST FAILED: llama-cli not on \$PATH" >&2; \
        echo "Searched for the binary at:" >&2; \
        find / -maxdepth 4 -name llama-cli -type f 2>/dev/null >&2 || true; \
        exit 1; \
    }; \
    command -v script >/dev/null 2>&1 || { \
        echo "BUILD SMOKE TEST FAILED: util-linux 'script' not available (needed for pty wrapping)" >&2; \
        exit 1; \
    }; \
    command -v jq >/dev/null 2>&1 || { \
        echo "BUILD SMOKE TEST FAILED: jq not available" >&2; \
        exit 1; \
    }; \
    echo "smoke test passed: llama-cli=$(command -v llama-cli) script=$(command -v script) jq=$(command -v jq)"

ENTRYPOINT ["/app/entrypoint.sh"]
