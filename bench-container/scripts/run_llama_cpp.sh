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
# -st (--single-turn): with --prompt set, runs one generation and exits.
#
# We wrap the invocation in `script -qfc ... /dev/null` to allocate a
# pseudo-tty. Recent llama-cli builds detect isatty() and suppress the
# banner, generated tokens, and the "[ Prompt: X t/s | Generation: Y t/s ]"
# timing summary when stdout is a pipe — which is exactly what happens
# when subprocess.run captures output. With a pty in the middle, llama-cli
# writes the full interactive UI and `script` forwards it to our stdout
# where the orchestrator's parser can see it.
#
# printf '%q' shell-escapes the prompt so `script -c` can eval it safely.
LLAMA_ARGS=(
    llama-cli
    -m "$GGUF_FILE"
    -p "$PROMPT"
    -n "$MAX_TOKENS"
    --temp "$TEMPERATURE"
    --top-p "$TOP_P"
    -ngl 999
    -st
)
QUOTED_CMD=$(printf '%q ' "${LLAMA_ARGS[@]}")
script -qfc "$QUOTED_CMD" /dev/null 2>&1
