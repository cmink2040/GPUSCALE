#!/usr/bin/env bash
# Do NOT use set -e — we handle errors per-iteration so one failure doesn't kill everything
set -uo pipefail

# All Python calls go through uv run to use the correct venv (Python 3.14 + CUDA PyTorch)
PY="uv run --project /app python"

# =============================================================================
# GPUSCALE Benchmark Container Entrypoint
#
# Required env vars:
#   MODEL            - Model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)
#   ENGINE           - "llama.cpp" or "vllm"
#
# Optional env vars:
#   WORKLOAD_CONFIG  - JSON string of workload spec (defaults to built-in)
#   MODEL_FORMAT     - "full", "gguf", "gptq" (auto-detected if not set)
#   GGUF_QUANT       - e.g. "Q4_K_M" (required for gguf)
#   VOLUME_MOUNT_PATH - persistent volume mount (e.g. /workspace, /runpod-volume)
#
# For S3 models:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT, S3_BUCKET, S3_MODEL_KEY
#
# For gated HuggingFace models:
#   HF_TOKEN
# =============================================================================

echo "=== GPUSCALE Benchmark Container ===" >&2
echo "MODEL:  ${MODEL:?MODEL env var is required}" >&2
echo "ENGINE: ${ENGINE:?ENGINE env var is required}" >&2

# Use persistent storage if available, otherwise ephemeral /models
MODEL_DIR="/models"
# Check for volume mount: explicit env var, or RunPod's default /workspace, or /runpod-volume
for CANDIDATE in "${VOLUME_MOUNT_PATH:-}" "/workspace" "/runpod-volume"; do
    if [ -n "$CANDIDATE" ] && [ -d "$CANDIDATE" ] && [ -w "$CANDIDATE" ]; then
        MODEL_DIR="${CANDIDATE}/models/${MODEL}/${MODEL_FORMAT:-full}"
        echo "Using persistent model dir: $MODEL_DIR" >&2
        break
    fi
done
export MODEL_DIR
mkdir -p "$MODEL_DIR"

# Default workload if not provided
if [ -z "${WORKLOAD_CONFIG:-}" ]; then
    WORKLOAD_CONFIG=$(cat /app/workloads/default.json)
    echo "Using default workload." >&2
fi

# Parse workload config
ITERATIONS=$(echo "$WORKLOAD_CONFIG" | jq -r '.iterations // 5')
WARMUP_ITERATIONS=$(echo "$WORKLOAD_CONFIG" | jq -r '.warmup_iterations // 1')
MAX_TOKENS=$(echo "$WORKLOAD_CONFIG" | jq -r '.generation_params.max_tokens // 512')
TEMPERATURE=$(echo "$WORKLOAD_CONFIG" | jq -r '.generation_params.temperature // 0.0')
TOP_P=$(echo "$WORKLOAD_CONFIG" | jq -r '.generation_params.top_p // 1.0')
NUM_PROMPTS=$(echo "$WORKLOAD_CONFIG" | jq '.prompts | length')
TOTAL_ITERATIONS=$((WARMUP_ITERATIONS + ITERATIONS))
TOTAL_RUNS=$(( TOTAL_ITERATIONS * NUM_PROMPTS ))
echo "Prompts: $NUM_PROMPTS, Iterations: $ITERATIONS (+ $WARMUP_ITERATIONS warmup), Total runs: $TOTAL_RUNS" >&2
echo "Max tokens: $MAX_TOKENS" >&2

# ---- Step 1: Pull model ----
echo "--- Pulling model ---" >&2
if ! $PY /app/scripts/pull_model.py; then
    echo "FATAL: Model pull failed. Cannot continue." >&2
    exit 1
fi
echo "--- Model ready ---" >&2

# Show what we got
echo "Model files:" >&2
ls -lh "$MODEL_DIR"/ 2>&1 | head -20 >&2

# Detect actual model format from files on disk (overrides whatever was passed)
HAS_GGUF=false
HAS_PTH=false
HAS_HF=false
ls "$MODEL_DIR"/*.gguf >/dev/null 2>&1 && HAS_GGUF=true
ls "$MODEL_DIR"/*.pth >/dev/null 2>&1 && HAS_PTH=true
[ -f "$MODEL_DIR/config.json" ] && HAS_HF=true

if [ "$HAS_GGUF" = true ]; then
    MODEL_FORMAT="gguf"
elif [ "$HAS_PTH" = true ]; then
    MODEL_FORMAT="pth"
elif [ "$HAS_HF" = true ]; then
    MODEL_FORMAT="hf"
elif [ -f "$MODEL_DIR/quantize_config.json" ]; then
    MODEL_FORMAT="gptq"
fi
echo "Detected model format: $MODEL_FORMAT" >&2

# Route engine based on what we actually have
if [ "$HAS_PTH" = true ]; then
    if [ "$ENGINE" != "meta-native" ]; then
        echo "Meta .pth format detected — switching engine from $ENGINE to meta-native." >&2
        ENGINE="meta-native"
    fi
elif [ "$HAS_GGUF" = true ] && [ "$ENGINE" != "llama.cpp" ]; then
    echo "GGUF format detected — switching engine to llama.cpp." >&2
    ENGINE="llama.cpp"
fi
echo "Using engine: $ENGINE" >&2

# ---- Step 2: Start GPU metrics collection ----
echo "--- Starting GPU metrics collection ---" >&2
/app/scripts/collect_metrics.sh start

# ---- Step 3: Run benchmark ----
echo "--- Running benchmark ($ENGINE, $TOTAL_ITERATIONS iterations x $NUM_PROMPTS prompts) ---" >&2

ERRORS=0
SUCCESSES=0

for i in $(seq 1 "$TOTAL_ITERATIONS"); do
    if [ "$i" -le "$WARMUP_ITERATIONS" ]; then
        ITER_LABEL="Warmup iteration $i/$WARMUP_ITERATIONS"
    else
        REAL_ITER=$((i - WARMUP_ITERATIONS))
        ITER_LABEL="Iteration $REAL_ITER/$ITERATIONS"
    fi

    for p in $(seq 0 $((NUM_PROMPTS - 1))); do
        PROMPT=$(echo "$WORKLOAD_CONFIG" | jq -r ".prompts[$p].content")
        PROMPT_LEN=${#PROMPT}
        echo "--- $ITER_LABEL, prompt $((p+1))/$NUM_PROMPTS ($PROMPT_LEN chars) ---" >&2

        case "$ENGINE" in
            llama.cpp)
                if /app/scripts/run_llama_cpp.sh "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P"; then
                    SUCCESSES=$((SUCCESSES + 1))
                else
                    echo "WARNING: llama.cpp failed on $ITER_LABEL prompt $((p+1)), continuing..." >&2
                    ERRORS=$((ERRORS + 1))
                fi
                ;;
            vllm)
                if $PY /app/scripts/run_vllm.py "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P"; then
                    SUCCESSES=$((SUCCESSES + 1))
                else
                    echo "WARNING: vLLM failed on $ITER_LABEL prompt $((p+1)), continuing..." >&2
                    ERRORS=$((ERRORS + 1))
                fi
                ;;
            meta-native)
                if $PY /app/scripts/run_meta_native.py "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P"; then
                    SUCCESSES=$((SUCCESSES + 1))
                else
                    echo "WARNING: Meta native failed on $ITER_LABEL prompt $((p+1)), continuing..." >&2
                    ERRORS=$((ERRORS + 1))
                fi
                ;;
            *)
                echo "ERROR: Unknown engine: $ENGINE" >&2
                ERRORS=$((ERRORS + 1))
                ;;
        esac
    done
done

echo "--- Benchmark complete: $SUCCESSES succeeded, $ERRORS failed ---" >&2

# ---- Step 4: Stop GPU metrics, emit result ----
echo "--- Stopping GPU metrics collection ---" >&2
/app/scripts/collect_metrics.sh stop

# Print the nvidia-smi CSV data so the orchestrator can parse it
echo "=== GPU_METRICS_START ==="
cat /tmp/gpu_metrics.csv 2>/dev/null || true
echo "=== GPU_METRICS_END ==="

# Exit 0 even if some iterations failed — partial results are still useful
exit 0
