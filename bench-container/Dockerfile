# =============================================================================
# GPUSCALE Benchmark Container
# Runs LLM inference benchmarks with llama.cpp or vLLM, collects GPU metrics.
# NO secrets are baked in — all credentials via runtime env vars.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build llama.cpp from source
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Pin to a specific release for reproducibility
ARG LLAMA_CPP_TAG=b5200
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} \
    https://github.com/ggerganov/llama.cpp /llama.cpp

WORKDIR /llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90" \
    && cmake --build build --config Release -j$(nproc)

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    jq curl \
    && rm -rf /var/lib/apt/lists/*

# Copy llama.cpp binaries from builder
COPY --from=builder /llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Install Python dependencies
# vLLM is large — install separately so Docker layer caching helps
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
RUN pip3 install --no-cache-dir vllm==0.7.3

# Copy application code
WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
COPY scripts/ /app/scripts/
COPY workloads/ /app/workloads/

RUN chmod +x /app/entrypoint.sh /app/scripts/*.sh

# Models will be downloaded here at runtime
RUN mkdir -p /models /results

# No secrets in the image — pass via: docker run -e AWS_ACCESS_KEY_ID=... etc.
ENTRYPOINT ["/app/entrypoint.sh"]
