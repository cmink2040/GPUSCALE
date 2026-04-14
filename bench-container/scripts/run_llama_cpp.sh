#!/usr/bin/env bash
set -uo pipefail
# Note: we deliberately do NOT use `set -e` here. We want full control over
# how the exit code from llama-cli propagates out, because we're wrapping it
# in `script -qfc` (which always returns 0, see below).

# Usage: run_llama_cpp.sh <prompt> <max_tokens> <temperature> <top_p>

PROMPT="$1"
MAX_TOKENS="${2:-512}"
TEMPERATURE="${3:-0.0}"
TOP_P="${4:-1.0}"

MODEL_DIR="${MODEL_DIR:-/models}"

# -----------------------------------------------------------------------------
# Locate llama-cli — agnostic to base image layout.
#
# Historically this script just called `llama-cli` as a bareword. That worked
# on ghcr.io/ggml-org/llama.cpp:full (binaries in /usr/local/bin) and broke
# silently on :light-cuda / :light-cuda13 (binaries at /app/llama-cli, not on
# $PATH). The Dockerfile now puts /app on PATH, and this loop is the second
# line of defense so the script keeps working across any future layout swap.
#
# Override by exporting LLAMA_BIN=/path/to/llama-cli.
# -----------------------------------------------------------------------------
LLAMA_BIN="${LLAMA_BIN:-}"
if [ -z "$LLAMA_BIN" ] || [ ! -x "$LLAMA_BIN" ]; then
    if command -v llama-cli >/dev/null 2>&1; then
        LLAMA_BIN=$(command -v llama-cli)
    else
        for candidate in \
            /app/llama-cli \
            /usr/local/bin/llama-cli \
            /usr/bin/llama-cli \
            /llama.cpp/llama-cli \
            /llama.cpp/build/bin/llama-cli; do
            if [ -x "$candidate" ]; then
                LLAMA_BIN="$candidate"
                break
            fi
        done
    fi
fi
if [ -z "$LLAMA_BIN" ] || [ ! -x "$LLAMA_BIN" ]; then
    echo "ERROR: llama-cli binary not found (searched \$PATH + /app, /usr/local/bin, /usr/bin, /llama.cpp, /llama.cpp/build/bin)" >&2
    exit 42
fi

# Find the GGUF file
GGUF_FILE=$(find "$MODEL_DIR" -name "*.gguf" -type f | head -1)

if [ -z "$GGUF_FILE" ]; then
    echo "ERROR: No .gguf file found in $MODEL_DIR/. llama.cpp requires GGUF format." >&2
    echo "Set MODEL_FORMAT=gguf and provide GGUF_QUANT, or use vllm for full-weight models." >&2
    exit 1
fi

echo "Using llama-cli: $LLAMA_BIN" >&2
echo "Using model:     $GGUF_FILE" >&2

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
# CRITICAL: `script -qfc "$CMD" /dev/null` always exits 0 when $CMD fails
# (script(1) only surfaces its own errors, not the child's). To avoid the
# "silent success" mode where llama-cli crashes and the harness records
# 0-metric rows, we embed a sentinel: the command string ends with
# `; printf $? > $STATUS_FILE`, run in the same inner shell. After script(1)
# returns, we read the sentinel and exit with the child's actual status.
LLAMA_ARGS=(
    "$LLAMA_BIN"
    -m "$GGUF_FILE"
    -p "$PROMPT"
    -n "$MAX_TOKENS"
    --temp "$TEMPERATURE"
    --top-p "$TOP_P"
    -ngl 999
    -st
)
STATUS_FILE=$(mktemp -t llama_status.XXXXXX)
trap 'rm -f "$STATUS_FILE"' EXIT

# printf '%q' shell-escapes the prompt so `script -c` can eval it safely.
QUOTED_CMD=$(printf '%q ' "${LLAMA_ARGS[@]}")
# Inside the script(1) arg, \$? stays literal and $STATUS_FILE is expanded
# by the outer shell — which is exactly what we want (the path is baked in,
# the child's exit status is read after it runs).
script -qfc "${QUOTED_CMD}; printf '%s' \$? > $STATUS_FILE" /dev/null 2>&1

CHILD_STATUS=$(cat "$STATUS_FILE" 2>/dev/null || true)
if [ -z "$CHILD_STATUS" ]; then
    echo "ERROR: llama-cli pty wrapper produced no exit status (script(1) itself failed?)" >&2
    exit 43
fi
exit "$CHILD_STATUS"
