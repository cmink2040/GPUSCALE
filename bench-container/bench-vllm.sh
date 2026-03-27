#!/usr/bin/env bash
# =============================================================================
# GPUSCALE vLLM Benchmark Script
# Mirrors entrypoint.sh but uses vLLM for inference on HuggingFace models.
#
# curl-bash into a vLLM-capable pod:
#   curl -sL https://raw.githubusercontent.com/cmink2040/GPUSCALE/main/bench-container/bench-vllm.sh | bash
#
# Required env vars:
#   MODEL            - HuggingFace model ID (e.g. Qwen/Qwen3.5-9B)
#
# Optional env vars:
#   MAX_TOKENS, TEMPERATURE, TOP_P, ITERATIONS, WARMUP
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT, S3_BUCKET - for result upload
#   GPUSCALE_RUN_ID  - unique run ID (auto-generated if not set)
#   HF_TOKEN         - for gated models
# =============================================================================

set -uo pipefail

echo "=== GPUSCALE vLLM Benchmark ===" >&2
echo "MODEL: ${MODEL:?MODEL env var is required}" >&2

# Config
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
ITERATIONS="${ITERATIONS:-3}"
WARMUP="${WARMUP:-1}"
RUN_ID="${GPUSCALE_RUN_ID:-vllm_$(hostname)_$(date +%s)}"
ENGINE_LOG="/tmp/gpuscale_engine.txt"
GPU_METRICS="/tmp/gpuscale_gpu.csv"
RESULT_FILE="/tmp/gpuscale_result.txt"

# Prompts — same standardized set as entrypoint.sh
PROMPTS=(
    "What is the capital of France?"
    "Explain the difference between TCP and UDP protocols. When would you use one over the other? Give concrete examples of applications that use each protocol and why that choice makes sense."
    "You are a senior software architect reviewing a system design. The system is a real-time bidding platform for digital advertising that needs to handle 500,000 requests per second with a p99 latency under 50ms. The current proposal uses a microservices architecture with the following components: 1. A load balancer distributing traffic across bid request handlers 2. A feature store (Redis cluster) for real-time user profile lookups 3. A machine learning inference service running bid prediction models 4. A Kafka-based event pipeline for logging and analytics 5. A PostgreSQL database for campaign management and budget tracking. The team is concerned about three issues: (a) the ML inference service adds 15-20ms of latency per request, (b) Redis cluster failover causes 2-3 second disruptions, and (c) the Kafka pipeline occasionally drops events during peak load. Provide a detailed technical review addressing each concern."
)
NUM_PROMPTS=${#PROMPTS[@]}
TOTAL_ITERATIONS=$((WARMUP + ITERATIONS))

echo "Prompts: $NUM_PROMPTS, Iterations: $ITERATIONS (+ $WARMUP warmup)" >&2
echo "Max tokens: $MAX_TOKENS" >&2

# ---- Step 1: Install vLLM if not present ----
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "--- Installing vLLM ---" >&2
    pip install -q vllm 2>&1 | tail -1 >&2
fi

# ---- Step 2: Start GPU metrics collection ----
echo "--- Starting GPU metrics collection ---" >&2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits -l 1 > "$GPU_METRICS" 2>/dev/null &
NVIDIA_PID=$!

# ---- Step 3: Run benchmark ----
echo "--- Running benchmark (vllm, $TOTAL_ITERATIONS iterations x $NUM_PROMPTS prompts) ---" >&2

> "$ENGINE_LOG"
ERRORS=0
SUCCESSES=0

# Write the benchmark runner inline — vLLM loads the model once, runs all iterations
python3 << 'PYEOF'
import os, sys, time, json

model = os.environ["MODEL"]
max_tokens = int(os.environ.get("MAX_TOKENS", "512"))
temperature = float(os.environ.get("TEMPERATURE", "0.0"))
top_p = float(os.environ.get("TOP_P", "1.0"))
iterations = int(os.environ.get("ITERATIONS", "3"))
warmup = int(os.environ.get("WARMUP", "1"))
engine_log = os.environ.get("ENGINE_LOG", "/tmp/gpuscale_engine.txt")
hf_token = os.environ.get("HF_TOKEN", None)

# Prompts
prompts = [
    "What is the capital of France?",
    "Explain the difference between TCP and UDP protocols. When would you use one over the other? Give concrete examples of applications that use each protocol and why that choice makes sense.",
    "You are a senior software architect reviewing a system design. The system is a real-time bidding platform for digital advertising that needs to handle 500,000 requests per second with a p99 latency under 50ms. The current proposal uses a microservices architecture with the following components: 1. A load balancer distributing traffic across bid request handlers 2. A feature store (Redis cluster) for real-time user profile lookups 3. A machine learning inference service running bid prediction models 4. A Kafka-based event pipeline for logging and analytics 5. A PostgreSQL database for campaign management and budget tracking. The team is concerned about three issues: (a) the ML inference service adds 15-20ms of latency per request, (b) Redis cluster failover causes 2-3 second disruptions, and (c) the Kafka pipeline occasionally drops events during peak load. Provide a detailed technical review addressing each concern.",
]

from vllm import LLM, SamplingParams

print(f"Loading model {model}...", file=sys.stderr)
llm = LLM(model=model, trust_remote_code=True, dtype="float16")
print("Model loaded.", file=sys.stderr)

sampling = SamplingParams(max_tokens=max_tokens, temperature=max(temperature, 0.01), top_p=top_p)
total = warmup + iterations
successes = 0
errors = 0

with open(engine_log, "w") as log:
    for i in range(1, total + 1):
        if i <= warmup:
            label = f"Warmup iteration {i}/{warmup}"
        else:
            label = f"Iteration {i - warmup}/{iterations}"

        for p_idx, prompt in enumerate(prompts):
            marker = f"--- {label}, prompt {p_idx+1}/{len(prompts)} ({len(prompt)} chars) ---"
            print(marker, file=sys.stderr)
            log.write(marker + "\n")

            try:
                start = time.perf_counter()
                outputs = llm.generate([prompt], sampling)
                end = time.perf_counter()

                output = outputs[0]
                num_tokens = len(output.outputs[0].token_ids)
                wall = end - start
                tps = num_tokens / wall if wall > 0 else 0

                ttft_ms = 0.0
                if hasattr(output, "metrics") and output.metrics:
                    if hasattr(output.metrics, "first_token_time") and output.metrics.first_token_time:
                        ttft_ms = output.metrics.first_token_time * 1000

                lines = [
                    f"Throughput: {tps:.2f} tokens/s",
                    f"TTFT: {ttft_ms:.2f} ms",
                    f"Total time: {wall:.2f} s",
                    f"Generated {num_tokens} tokens",
                ]
                for line in lines:
                    print(line)
                    log.write(line + "\n")
                successes += 1

            except Exception as e:
                print(f"WARNING: vLLM failed: {e}", file=sys.stderr)
                log.write(f"ERROR: {e}\n")
                errors += 1

print(f"successes={successes}", file=sys.stderr)
print(f"errors={errors}", file=sys.stderr)
PYEOF

echo "--- Benchmark complete ---" >&2

# ---- Step 4: Stop GPU metrics ----
kill $NVIDIA_PID 2>/dev/null; wait $NVIDIA_PID 2>/dev/null
echo "--- GPU metrics stopped ---" >&2

# ---- Step 5: Assemble and upload result ----
{
    echo "=== ENGINE_OUTPUT_START ==="
    cat "$ENGINE_LOG" 2>/dev/null
    echo "=== ENGINE_OUTPUT_END ==="
    echo "=== GPU_METRICS_START ==="
    cat "$GPU_METRICS" 2>/dev/null
    echo "=== GPU_METRICS_END ==="
    echo "=== BENCHMARK_SUMMARY ==="
    echo "engine=vllm"
    echo "model=$MODEL"
} > "$RESULT_FILE"

cat "$RESULT_FILE"

# Upload to S3
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
