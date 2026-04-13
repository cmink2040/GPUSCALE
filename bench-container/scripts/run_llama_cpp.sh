#!/usr/bin/env bash
set -euo pipefail

# Usage: run_llama_cpp.sh <prompt> <max_tokens> <temperature> <top_p>

PROMPT="$1"
MAX_TOKENS="${2:-512}"
TEMPERATURE="${3:-0.0}"
TOP_P="${4:-1.0}"

MODEL_DIR="${MODEL_DIR:-/models}"

# Find the GGUF file
GGUF_FILE=$(find "$MODEL_DIR" -name "*.gguf" -type f | head -1)

if [ -z "$GGUF_FILE" ]; then
    echo "ERROR: No .gguf file found in $MODEL_DIR/. llama.cpp requires GGUF format." >&2
    echo "Set MODEL_FORMAT=gguf and provide GGUF_QUANT, or use vllm for full-weight models." >&2
    exit 1
fi

echo "Using model: $GGUF_FILE" >&2

# Run llama.cpp inference
# -ngl 999: offload all layers to GPU
# -no-cnv: force one-shot mode. Recent llama-cli builds auto-enable
#   conversation mode when the model ships a chat template (e.g. Llama-3.1
#   Instruct), which drops into an interactive REPL and blocks forever on
#   stdin. We want a single generate-then-exit run.
# --no-display-prompt: don't echo the prompt back
llama-cli \
    -m "$GGUF_FILE" \
    -p "$PROMPT" \
    -n "$MAX_TOKENS" \
    --temp "$TEMPERATURE" \
    --top-p "$TOP_P" \
    -ngl 999 \
    -no-cnv \
    --no-display-prompt \
    </dev/null 2>&1
