#!/usr/bin/env python3
"""GPUSCALE CPU Benchmark — collects system info, runs vLLM on CPU, reports results.

Usage: Upload to a machine and run:
    python3 bench-cpu.py [--model MODEL] [--dtype bfloat16]

Env vars: VLLM_API_KEY, HF_TOKEN, S3_BUCKET, AWS_ACCESS_KEY_ID, etc.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
DTYPE = os.environ.get("DTYPE", "bfloat16")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "gpuscale-bench")
PORT = 8000
ITERATIONS = 3
WARMUP = 1

PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between TCP and UDP protocols. When would you use one over the other? Give concrete examples of applications that use each protocol and why that choice makes sense.",
    (
        "You are a senior software architect reviewing a system design. The system is a real-time bidding platform "
        "for digital advertising that needs to handle 500,000 requests per second with a p99 latency under 50ms. "
        "The current proposal uses a microservices architecture with the following components:\n\n"
        "1. A load balancer distributing traffic across bid request handlers\n"
        "2. A feature store (Redis cluster) for real-time user profile lookups\n"
        "3. A machine learning inference service running bid prediction models\n"
        "4. A Kafka-based event pipeline for logging and analytics\n"
        "5. A PostgreSQL database for campaign management and budget tracking\n\n"
        "Provide a detailed technical review addressing each concern."
    ),
]


# ---------------------------------------------------------------------------
# System info collection
# ---------------------------------------------------------------------------

def run_cmd(cmd: str, timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def collect_cpu_info() -> dict:
    info = {}

    # CPU model and cores
    lscpu = run_cmd("lscpu")
    for line in lscpu.split("\n"):
        if "Model name:" in line:
            info["cpu_model"] = line.split(":", 1)[1].strip()
        elif "Socket(s):" in line:
            info["cpu_sockets"] = int(line.split(":", 1)[1].strip())
        elif "Core(s) per socket:" in line:
            info["cores_per_socket"] = int(line.split(":", 1)[1].strip())
        elif "Thread(s) per core:" in line:
            info["threads_per_core"] = int(line.split(":", 1)[1].strip())
        elif "CPU max MHz:" in line:
            info["cpu_max_mhz"] = float(line.split(":", 1)[1].strip())
        elif "CPU MHz:" in line and "cpu_mhz" not in info:
            info["cpu_mhz"] = float(line.split(":", 1)[1].strip())
        elif "L3 cache:" in line:
            info["l3_cache"] = line.split(":", 1)[1].strip()
        elif "Flags:" in line:
            flags = line.split(":", 1)[1].strip()
            exts = []
            if "avx512" in flags:
                exts.append("AVX-512")
            if "avx2" in flags:
                exts.append("AVX2")
            if "avx " in flags:
                exts.append("AVX")
            if "amx" in flags:
                exts.append("AMX")
            info["cpu_extensions"] = ",".join(exts)

    info["cpu_cores"] = info.get("cpu_sockets", 1) * info.get("cores_per_socket", 0)
    info["cpu_threads"] = info["cpu_cores"] * info.get("threads_per_core", 1)

    # RAM info
    meminfo = run_cmd("cat /proc/meminfo | grep MemTotal")
    if meminfo:
        kb = int(re.search(r"(\d+)", meminfo).group(1))
        info["ram_total_gb"] = round(kb / 1024 / 1024, 1)

    # Try dmidecode for DIMM details (needs root)
    dimm_output = run_cmd("dmidecode --type 17 2>/dev/null")
    if dimm_output:
        dimm_count = 0
        dimm_speeds = []
        dimm_type = ""
        for line in dimm_output.split("\n"):
            line = line.strip()
            if line.startswith("Size:") and "No Module" not in line and "0 MB" not in line:
                dimm_count += 1
            elif line.startswith("Speed:") and "Unknown" not in line:
                speed_match = re.search(r"(\d+)", line)
                if speed_match:
                    dimm_speeds.append(int(speed_match.group(1)))
            elif line.startswith("Type:") and line != "Type: Unknown":
                t = line.split(":", 1)[1].strip()
                if t in ("DDR4", "DDR5", "DDR3"):
                    dimm_type = t
        info["ram_dimms"] = dimm_count
        info["ram_type"] = dimm_type
        if dimm_speeds:
            info["ram_speed_mhz"] = max(dimm_speeds)
    else:
        # Fallback — try to infer from /sys
        info["ram_dimms"] = 0
        info["ram_type"] = ""
        info["ram_speed_mhz"] = 0

    # NUMA topology
    numa = run_cmd("numactl --hardware 2>/dev/null | head -5")
    if numa:
        info["numa_nodes"] = numa.count("node ")

    # Measure actual memory bandwidth with mbw
    # Install if needed, run a quick test
    run_cmd("apt-get update -qq && apt-get install -y -qq mbw 2>/dev/null || pip install mbw 2>/dev/null", timeout=30)
    mbw_out = run_cmd("mbw -n 3 256 2>/dev/null | grep AVG", timeout=60)
    if mbw_out:
        # Parse: AVG	Method: MEMCPY	Elapsed: 0.12345	MiB: 256.000	Copy: 12345.678 MiB/s
        bw_values = []
        for line in mbw_out.split("\n"):
            match = re.search(r"Copy:\s+([\d.]+)\s+MiB/s", line)
            if match:
                bw_values.append(float(match.group(1)))
        if bw_values:
            info["measured_bandwidth_mibs"] = round(max(bw_values), 1)
            info["measured_bandwidth_gbs"] = round(max(bw_values) / 1024, 2)
            print(f"Measured bandwidth: {info['measured_bandwidth_gbs']} GB/s", file=sys.stderr)

    return info


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def start_vllm_server(model: str, dtype: str) -> subprocess.Popen:
    """Start vLLM in CPU mode."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--dtype", dtype,
        "--enforce-eager",
        "--max-model-len", "2048",
        "--api-key", VLLM_API_KEY,
    ]

    env = os.environ.copy()
    env["VLLM_TARGET_DEVICE"] = "cpu"
    env["VLLM_CPU_KVCACHE_SPACE"] = "8"  # 8GB KV cache

    log = open("/tmp/vllm_cpu.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
    print(f"vLLM server started (PID {proc.pid})", file=sys.stderr)
    return proc


def wait_for_server(proc: subprocess.Popen, timeout: int = 600) -> bool:
    """Wait for vLLM health endpoint."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print("vLLM process died!", file=sys.stderr)
            log = Path("/tmp/vllm_cpu.log").read_text()
            print(log[-1000:], file=sys.stderr)
            return False
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3)
            return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(model: str) -> list[dict]:
    """Run inference benchmark against the local vLLM server."""
    import urllib.request

    results = []
    total_iters = WARMUP + ITERATIONS

    for i in range(total_iters):
        is_warmup = i < WARMUP
        label = f"Warmup {i+1}/{WARMUP}" if is_warmup else f"Iteration {i-WARMUP+1}/{ITERATIONS}"

        for p_idx, prompt in enumerate(PROMPTS):
            tag = f"{label}, prompt {p_idx+1}/{len(PROMPTS)} ({len(prompt)} chars)"
            print(f"--- {tag} ---", file=sys.stderr)

            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0,
            }).encode()

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VLLM_API_KEY}",
            }

            req = urllib.request.Request(
                f"http://localhost:{PORT}/v1/chat/completions",
                data=payload, headers=headers,
            )

            start = time.perf_counter()
            try:
                resp = urllib.request.urlopen(req, timeout=300)
                data = json.loads(resp.read().decode())
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                continue
            elapsed = time.perf_counter() - start

            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            tps = tokens / elapsed if elapsed > 0 else 0

            print(f"Throughput: {tps:.2f} tokens/s")
            print(f"Total time: {elapsed:.2f} s")
            print(f"Generated {tokens} tokens")

            if not is_warmup:
                results.append({
                    "prompt_idx": p_idx,
                    "tokens": tokens,
                    "wall_time_s": elapsed,
                    "tokens_per_sec": tps,
                })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== GPUSCALE CPU Benchmark ===", file=sys.stderr)
    print(f"Model: {MODEL}", file=sys.stderr)
    print(f"Dtype: {DTYPE}", file=sys.stderr)

    # Collect system info
    print("Collecting system info...", file=sys.stderr)
    cpu_info = collect_cpu_info()
    print(f"CPU: {cpu_info.get('cpu_model', '?')}", file=sys.stderr)
    print(f"Cores: {cpu_info.get('cpu_cores', '?')} ({cpu_info.get('cpu_sockets', '?')} socket(s))", file=sys.stderr)
    print(f"RAM: {cpu_info.get('ram_total_gb', '?')}GB, {cpu_info.get('ram_dimms', '?')} DIMMs, {cpu_info.get('ram_type', '?')} @ {cpu_info.get('ram_speed_mhz', '?')}MHz", file=sys.stderr)
    print(f"Extensions: {cpu_info.get('cpu_extensions', '?')}", file=sys.stderr)

    # Start vLLM
    print("Starting vLLM server (CPU mode)...", file=sys.stderr)
    proc = start_vllm_server(MODEL, DTYPE)

    print("Waiting for server...", file=sys.stderr)
    if not wait_for_server(proc):
        print("FATAL: vLLM server failed to start", file=sys.stderr)
        proc.kill()
        sys.exit(1)
    print("vLLM ready!", file=sys.stderr)

    # Run benchmark
    print("=== ENGINE_OUTPUT_START ===")
    results = run_benchmark(MODEL)
    print("=== ENGINE_OUTPUT_END ===")

    # Compute aggregates
    if results:
        tps_values = [r["tokens_per_sec"] for r in results]
        avg_tps = sum(tps_values) / len(tps_values)
        total_wall = sum(r["wall_time_s"] for r in results)
    else:
        avg_tps = 0
        total_wall = 0

    # GPU metrics placeholder (no GPU)
    print("=== GPU_METRICS_START ===")
    print("=== GPU_METRICS_END ===")

    # Summary
    print("=== BENCHMARK_SUMMARY ===")
    print(f"engine=vllm-cpu")
    print(f"model={MODEL}")
    print(f"dtype={DTYPE}")
    print(f"avg_tps={avg_tps:.2f}")
    print(f"total_wall_time={total_wall:.2f}")

    # Full result with CPU info
    full_result = {
        "cpu_info": cpu_info,
        "model": MODEL,
        "dtype": DTYPE,
        "engine": "vllm-cpu",
        "iterations": results,
        "avg_tokens_per_sec": round(avg_tps, 2),
        "total_wall_time_s": round(total_wall, 2),
    }
    print("=== CPU_RESULT_JSON ===")
    print(json.dumps(full_result))
    print("=== CPU_RESULT_JSON_END ===")

    # Cleanup
    proc.kill()
    print("=== CPU BENCHMARK COMPLETE ===", file=sys.stderr)


if __name__ == "__main__":
    main()
