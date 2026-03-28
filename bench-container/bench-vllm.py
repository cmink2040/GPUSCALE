#!/usr/bin/env python3
"""GPUSCALE vLLM Benchmark — runs against a local vLLM server on localhost:8000.

curl-bash usage:
  curl -sL https://raw.githubusercontent.com/cmink2040/GPUSCALE/main/bench-container/bench-vllm.py | python3
"""

import json
import os
import subprocess
import sys
import time

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
ITERATIONS = int(os.environ.get("ITERATIONS", "3"))
WARMUP = int(os.environ.get("WARMUP", "1"))
RUN_ID = os.environ.get("GPUSCALE_RUN_ID", f"vllm_{int(time.time())}")

PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between TCP and UDP protocols. When would you use one over the other? Give concrete examples of applications that use each protocol and why that choice makes sense.",
    "You are a senior software architect reviewing a system design. The system is a real-time bidding platform for digital advertising that needs to handle 500,000 requests per second with a p99 latency under 50ms. The current proposal uses a microservices architecture with the following components: 1. A load balancer distributing traffic across bid request handlers 2. A feature store Redis cluster for real-time user profile lookups 3. A machine learning inference service running bid prediction models 4. A Kafka-based event pipeline for logging and analytics 5. A PostgreSQL database for campaign management and budget tracking. The team is concerned about three issues: a the ML inference service adds 15-20ms of latency per request, b Redis cluster failover causes 2-3 second disruptions, and c the Kafka pipeline occasionally drops events during peak load. Provide a detailed technical review addressing each concern.",
]

RESULT_FILE = "/tmp/gpuscale_result.txt"
ENGINE_LOG = "/tmp/gpuscale_engine.txt"
GPU_METRICS = "/tmp/gpuscale_gpu.csv"


def wait_for_server():
    """Wait for vLLM server to be ready."""
    print("Waiting for vLLM server...", file=sys.stderr)
    for i in range(120):
        try:
            import urllib.request
            req = urllib.request.Request(f"{VLLM_URL}/v1/models")
            if VLLM_API_KEY:
                req.add_header("Authorization", f"Bearer {VLLM_API_KEY}")
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            model_name = data["data"][0]["id"]
            print(f"vLLM ready. Model: {model_name}", file=sys.stderr)
            return model_name
        except Exception:
            if i % 12 == 0:
                print(f"  Waiting... ({i*5}s)", file=sys.stderr)
            time.sleep(5)
    print("FATAL: vLLM server not ready after 10 minutes.", file=sys.stderr)
    sys.exit(1)


def call_vllm(model_name, prompt):
    """Make a chat completion request and return (tokens, wall_time_s, ttft_s)."""
    import urllib.request

    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.01,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{VLLM_URL}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}" if VLLM_API_KEY else "",
        },
    )

    start = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
    except Exception as e:
        print(f"  Request failed: {e}", file=sys.stderr)
        return 0, 0, 0
    end = time.perf_counter()

    tokens = data.get("usage", {}).get("completion_tokens", 0)
    wall = end - start
    return tokens, wall, 0  # TTFT not available from non-streaming


def main():
    print("=== GPUSCALE vLLM Benchmark ===", file=sys.stderr)
    print(f"Run ID: {RUN_ID}", file=sys.stderr)
    print(f"Iterations: {ITERATIONS} (+ {WARMUP} warmup)", file=sys.stderr)

    model_name = wait_for_server()

    # Start nvidia-smi polling
    nvidia_proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits", "-l", "1"],
        stdout=open(GPU_METRICS, "w"),
        stderr=subprocess.DEVNULL,
    )

    # Run benchmark
    total = WARMUP + ITERATIONS
    results = []

    with open(ENGINE_LOG, "w") as log:
        for i in range(1, total + 1):
            label = f"Warmup iteration {i}/{WARMUP}" if i <= WARMUP else f"Iteration {i - WARMUP}/{ITERATIONS}"

            for p_idx, prompt in enumerate(PROMPTS):
                marker = f"--- {label}, prompt {p_idx+1}/{len(PROMPTS)} ({len(prompt)} chars) ---"
                print(marker, file=sys.stderr)
                log.write(marker + "\n")

                tokens, wall, ttft = call_vllm(model_name, prompt)
                tps = tokens / wall if wall > 0 else 0

                lines = [
                    f"Throughput: {tps:.2f} tokens/s",
                    f"TTFT: {ttft*1000:.2f} ms",
                    f"Total time: {wall:.2f} s",
                    f"Generated {tokens} tokens",
                ]
                for line in lines:
                    print(line)
                    log.write(line + "\n")
                    log.flush()

    # Stop nvidia-smi
    nvidia_proc.terminate()
    nvidia_proc.wait()

    # Assemble result
    with open(RESULT_FILE, "w") as f:
        f.write("=== ENGINE_OUTPUT_START ===\n")
        f.write(open(ENGINE_LOG).read())
        f.write("=== ENGINE_OUTPUT_END ===\n")
        f.write("=== GPU_METRICS_START ===\n")
        try:
            f.write(open(GPU_METRICS).read())
        except Exception:
            pass
        f.write("=== GPU_METRICS_END ===\n")
        f.write("=== BENCHMARK_SUMMARY ===\n")
        f.write(f"engine=vllm\n")
        f.write(f"model={model_name}\n")

    # Print result to stdout
    print(open(RESULT_FILE).read())

    # Upload to S3
    bucket = os.environ.get("S3_BUCKET", "")
    if bucket and os.environ.get("AWS_ACCESS_KEY_ID"):
        try:
            import boto3
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "boto3"], capture_output=True)
            import boto3

        s3_key = f"results/{RUN_ID}.txt"
        print(f"Uploading to s3://{bucket}/{s3_key}", file=sys.stderr)
        try:
            client = boto3.client("s3",
                endpoint_url=os.environ.get("S3_ENDPOINT", ""),
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
            client.upload_file(RESULT_FILE, bucket, s3_key)
            print(f"Uploaded.", file=sys.stderr)
        except Exception as e:
            print(f"S3 upload failed: {e}", file=sys.stderr)

    print("=== Done ===", file=sys.stderr)


if __name__ == "__main__":
    main()
