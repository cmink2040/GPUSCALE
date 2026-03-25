#!/usr/bin/env bash
set -euo pipefail

# Usage: run_llama_cpp.sh <prompt> <max_tokens> <temperature> <top_p>

PROMPT="$1"
MAX_TOKENS="${2:-512}"
TEMPERATURE="${3:-0.0}"
TOP_P="${4:-1.0}"

# Find the GGUF file
GGUF_FILE=$(find /models -name "*.gguf" -type f | head -1)

if [ -z "$GGUF_FILE" ]; then
    echo "ERROR: No .gguf file found in /models/. llama.cpp requires GGUF format." >&2
    echo "Set MODEL_FORMAT=gguf and provide GGUF_QUANT, or use vllm for full-weight models." >&2
    exit 1
fi

echo "Using model: $GGUF_FILE" >&2

# Run llama.cpp inference
# -ngl 999: offload all layers to GPU
# --no-display-prompt: don't echo the prompt back
# The timing output goes to stderr (llama_print_timings)
# which the orchestrator's regex parser expects
llama-cli \
    -m "$GGUF_FILE" \
    -p "$PROMPT" \
    -n "$MAX_TOKENS" \
    --temp "$TEMPERATURE" \
    --top-p "$TOP_P" \
    -ngl 999 \
    --no-display-prompt \
    2>&1
