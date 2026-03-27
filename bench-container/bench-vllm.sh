#!/usr/bin/env bash
# =============================================================================
# GPUSCALE vLLM Benchmark Script
# Runs against a local vLLM server (localhost:8000) already serving a model.
#
# curl-bash into a running vLLM pod:
#   curl -sL https://raw.githubusercontent.com/cmink2040/GPUSCALE/main/bench-container/bench-vllm.sh | bash
#
# Required: vLLM server running on localhost:8000
# Optional env vars:
#   MAX_TOKENS, ITERATIONS, WARMUP
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT, S3_BUCKET
#   GPUSCALE_RUN_ID
# =============================================================================

set -uo pipefail

VLLM_URL="http://localhost:8000"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ITERATIONS="${ITERATIONS:-3}"
WARMUP="${WARMUP:-1}"
RUN_ID="${GPUSCALE_RUN_ID:-vllm_$(hostname)_$(date +%s)}"
ENGINE_LOG="/tmp/gpuscale_engine.txt"
GPU_METRICS="/tmp/gpuscale_gpu.csv"
RESULT_FILE="/tmp/gpuscale_result.txt"

echo "=== GPUSCALE vLLM Benchmark ===" >&2
echo "Run ID: $RUN_ID" >&2

# ---- Wait for vLLM server ----
echo "--- Waiting for vLLM server at $VLLM_URL ---" >&2
for i in $(seq 1 120); do
    if curl -s "$VLLM_URL/v1/models" >/dev/null 2>&1; then
        echo "vLLM server ready." >&2
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "FATAL: vLLM server not ready after 10 minutes." >&2
        exit 1
    fi
    sleep 5
done

# Get model name from server
MODEL_NAME=$(curl -s "$VLLM_URL/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
echo "Model: $MODEL_NAME" >&2

# ---- Prompts ----
# Same standardized set as entrypoint.sh
PROMPT_SHORT="What is the capital of France?"
PROMPT_MED="Explain the difference between TCP and UDP protocols. When would you use one over the other? Give concrete examples of applications that use each protocol and why that choice makes sense."
PROMPT_LONG="You are a senior software architect reviewing a system design. The system is a real-time bidding platform for digital advertising that needs to handle 500,000 requests per second with a p99 latency under 50ms. The current proposal uses a microservices architecture with the following components: 1. A load balancer distributing traffic across bid request handlers 2. A feature store Redis cluster for real-time user profile lookups 3. A machine learning inference service running bid prediction models 4. A Kafka-based event pipeline for logging and analytics 5. A PostgreSQL database for campaign management and budget tracking. The team is concerned about three issues: a the ML inference service adds 15-20ms of latency per request, b Redis cluster failover causes 2-3 second disruptions, and c the Kafka pipeline occasionally drops events during peak load. Provide a detailed technical review addressing each concern."

PROMPTS=("$PROMPT_SHORT" "$PROMPT_MED" "$PROMPT_LONG")
NUM_PROMPTS=${#PROMPTS[@]}
TOTAL=$((WARMUP + ITERATIONS))

echo "Prompts: $NUM_PROMPTS, Iterations: $ITERATIONS (+ $WARMUP warmup)" >&2
echo "Max tokens: $MAX_TOKENS" >&2

# ---- Start GPU metrics ----
echo "--- Starting GPU metrics ---" >&2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits -l 1 > "$GPU_METRICS" 2>/dev/null &
NVIDIA_PID=$!

# ---- Run benchmark ----
> "$ENGINE_LOG"
ERRORS=0
SUCCESSES=0

for i in $(seq 1 "$TOTAL"); do
    if [ "$i" -le "$WARMUP" ]; then
        ITER_LABEL="Warmup iteration $i/$WARMUP"
    else
        REAL=$((i - WARMUP))
        ITER_LABEL="Iteration $REAL/$ITERATIONS"
    fi

    for p_idx in $(seq 0 $((NUM_PROMPTS - 1))); do
        PROMPT="${PROMPTS[$p_idx]}"
        PROMPT_LEN=${#PROMPT}
        echo "--- $ITER_LABEL, prompt $((p_idx+1))/$NUM_PROMPTS ($PROMPT_LEN chars) ---" >&2
        echo "--- $ITER_LABEL, prompt $((p_idx+1))/$NUM_PROMPTS ---" >> "$ENGINE_LOG"

        # Time the request to localhost
        START_TIME=$(python3 -c "import time; print(time.time())")

        RESPONSE=$(curl -s -w '\n%{time_starttransfer}' \
            "$VLLM_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL_NAME\",
                \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json; print(json.dumps('$PROMPT'))")}],
                \"max_tokens\": $MAX_TOKENS,
                \"temperature\": 0.01,
                \"stream\": false
            }" 2>/dev/null)

        END_TIME=$(python3 -c "import time; print(time.time())")

        # Parse
        PARSED=$(python3 << PYEOF
import json, sys
raw = """$RESPONSE"""
lines = raw.strip().split('\n')
ttft_s = float(lines[-1]) if len(lines) > 1 else 0
body = '\n'.join(lines[:-1])
try:
    d = json.loads(body)
    usage = d.get('usage', {})
    tokens = usage.get('completion_tokens', 0)
    wall = $END_TIME - $START_TIME
    tps = tokens / wall if wall > 0 else 0
    ttft_ms = ttft_s * 1000
    print(f'Throughput: {tps:.2f} tokens/s')
    print(f'TTFT: {ttft_ms:.2f} ms')
    print(f'Total time: {wall:.2f} s')
    print(f'Generated {tokens} tokens')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
PYEOF
        )

        if [ $? -eq 0 ]; then
            echo "$PARSED" | tee -a "$ENGINE_LOG"
            SUCCESSES=$((SUCCESSES + 1))
        else
            echo "WARNING: Failed on $ITER_LABEL prompt $((p_idx+1))" >&2
            ERRORS=$((ERRORS + 1))
        fi
    done
done

echo "--- Benchmark complete: $SUCCESSES succeeded, $ERRORS failed ---" >&2

# ---- Stop GPU metrics ----
kill $NVIDIA_PID 2>/dev/null; wait $NVIDIA_PID 2>/dev/null

# ---- Assemble result ----
{
    echo "=== ENGINE_OUTPUT_START ==="
    cat "$ENGINE_LOG"
    echo "=== ENGINE_OUTPUT_END ==="
    echo "=== GPU_METRICS_START ==="
    cat "$GPU_METRICS" 2>/dev/null
    echo "=== GPU_METRICS_END ==="
    echo "=== BENCHMARK_SUMMARY ==="
    echo "successes=$SUCCESSES"
    echo "errors=$ERRORS"
    echo "engine=vllm"
    echo "model=$MODEL_NAME"
} > "$RESULT_FILE"

cat "$RESULT_FILE"

# ---- Upload to S3 ----
if [ -n "${S3_BUCKET:-}" ] && [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
    RESULT_S3_KEY="results/${RUN_ID}.txt"
    echo "--- Uploading to s3://${S3_BUCKET}/${RESULT_S3_KEY} ---" >&2
    pip install -q boto3 2>/dev/null
    python3 -c "
import os, boto3
client = boto3.client('s3',
    endpoint_url=os.environ.get('S3_ENDPOINT', ''),
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
)
client.upload_file('$RESULT_FILE', os.environ['S3_BUCKET'], '$RESULT_S3_KEY')
print(f'Uploaded to s3://{os.environ[\"S3_BUCKET\"]}/$RESULT_S3_KEY')
" 2>&1 || echo "WARNING: S3 upload failed" >&2
fi

echo "=== Done ===" >&2
