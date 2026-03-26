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

# Python deps — llama-models for Meta native .pth, boto3/HF for model pulling
COPY requirements.llamacpp.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

# Application code
WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
COPY scripts/ /app/scripts/
COPY workloads/ /app/workloads/

RUN chmod +x /app/entrypoint.sh /app/scripts/*.sh
RUN mkdir -p /models /results

ENTRYPOINT ["/app/entrypoint.sh"]
