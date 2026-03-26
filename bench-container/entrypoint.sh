#!/usr/bin/env bash
set -euo pipefail

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
# Count prompts
NUM_PROMPTS=$(echo "$WORKLOAD_CONFIG" | jq '.prompts | length')
TOTAL_ITERATIONS=$((WARMUP_ITERATIONS + ITERATIONS))
TOTAL_RUNS=$(( TOTAL_ITERATIONS * NUM_PROMPTS ))
echo "Prompts: $NUM_PROMPTS, Iterations: $ITERATIONS (+ $WARMUP_ITERATIONS warmup), Total runs: $TOTAL_RUNS" >&2
echo "Max tokens: $MAX_TOKENS" >&2

# ---- Step 1: Pull model ----
echo "--- Pulling model ---" >&2
python3 /app/scripts/pull_model.py
echo "--- Model ready ---" >&2

# Auto-detect model format if not set
if [ -z "${MODEL_FORMAT:-}" ]; then
    if ls /models/*.gguf >/dev/null 2>&1; then
        MODEL_FORMAT="gguf"
    elif [ -f /models/quantize_config.json ]; then
        MODEL_FORMAT="gptq"
    else
        MODEL_FORMAT="full"
    fi
    echo "Auto-detected format: $MODEL_FORMAT" >&2
fi

# ---- Step 2: Start GPU metrics collection ----
echo "--- Starting GPU metrics collection ---" >&2
/app/scripts/collect_metrics.sh start

# ---- Step 3: Run benchmark ----
echo "--- Running benchmark ($ENGINE, $TOTAL_ITERATIONS iterations) ---" >&2

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
                /app/scripts/run_llama_cpp.sh "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P"
                ;;
            vllm)
                python3 /app/scripts/run_vllm.py "$PROMPT" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P"
                ;;
            *)
                echo "ERROR: Unknown engine: $ENGINE" >&2
                exit 1
                ;;
        esac
    done
done

# ---- Step 4: Stop GPU metrics, emit result ----
echo "--- Stopping GPU metrics collection ---" >&2
/app/scripts/collect_metrics.sh stop

# Print the nvidia-smi CSV data so the orchestrator can parse it
echo "=== GPU_METRICS_START ==="
cat /tmp/gpu_metrics.csv 2>/dev/null || true
echo "=== GPU_METRICS_END ==="
