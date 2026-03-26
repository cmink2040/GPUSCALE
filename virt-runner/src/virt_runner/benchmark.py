"""Orchestrate benchmark runs: run container, parse combined output, collect metrics."""

from __future__ import annotations

import re
import subprocess
import time
import uuid

from rich.console import Console

from virt_runner.config import JobConfig
from virt_runner.metrics import (
    MetricsCollector,
    parse_llamacpp_output,
    parse_nvidia_smi_output,
    parse_vllm_output,
)
from virt_runner.models import BenchmarkResult, InferenceEngine, WorkloadConfig
from virt_runner.providers.base import ProvisionedInstance

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Output parsing: split engine output from GPU metrics block
# ---------------------------------------------------------------------------

ENGINE_OUTPUT_START = "=== ENGINE_OUTPUT_START ==="
ENGINE_OUTPUT_END = "=== ENGINE_OUTPUT_END ==="
GPU_METRICS_START = "=== GPU_METRICS_START ==="
GPU_METRICS_END = "=== GPU_METRICS_END ==="


def split_container_output(raw: str) -> tuple[str, str]:
    """Split the combined container output into engine output and GPU metrics CSV.

    The result file has sections delimited by markers:
      === ENGINE_OUTPUT_START === ... === ENGINE_OUTPUT_END ===
      === GPU_METRICS_START === ... === GPU_METRICS_END ===

    Returns (engine_output, gpu_metrics_csv).
    """
    # Extract engine output
    engine_output = ""
    eng_start = raw.find(ENGINE_OUTPUT_START)
    eng_end = raw.find(ENGINE_OUTPUT_END)
    if eng_start != -1 and eng_end != -1:
        engine_output = raw[eng_start + len(ENGINE_OUTPUT_START) : eng_end].strip()
    else:
        # Fallback: everything before GPU_METRICS_START is engine output
        gpu_start = raw.find(GPU_METRICS_START)
        if gpu_start != -1:
            engine_output = raw[:gpu_start].rstrip()
        else:
            engine_output = raw

    # Extract GPU metrics
    gpu_csv = ""
    gpu_start = raw.find(GPU_METRICS_START)
    gpu_end = raw.find(GPU_METRICS_END)
    if gpu_start != -1 and gpu_end != -1:
        gpu_csv = raw[gpu_start + len(GPU_METRICS_START) : gpu_end].strip()

    return engine_output, gpu_csv


def _parse_engine_output(engine: InferenceEngine, output: str):
    """Route engine output to the appropriate parser."""
    if engine == InferenceEngine.LLAMA_CPP:
        return parse_llamacpp_output(output)
    elif engine == InferenceEngine.VLLM:
        return parse_vllm_output(output)
    else:
        # meta-native uses same output format as vllm (Throughput/TTFT/Total time)
        return parse_vllm_output(output)


# ---------------------------------------------------------------------------
# Local benchmark
# ---------------------------------------------------------------------------


def run_local_benchmark(
    config: JobConfig,
    instance: ProvisionedInstance,
    workload: WorkloadConfig,
) -> BenchmarkResult:
    """Run the benchmark container locally with `docker run --gpus all`.

    The container handles all iterations, warmup, model pulling, and metrics
    collection internally. We capture its stdout (which contains engine timing
    lines interleaved with iteration markers, followed by the GPU_METRICS block).
    """
    run_id = f"local-{uuid.uuid4().hex[:12]}"
    result = BenchmarkResult(
        run_id=run_id,
        provider=config.provider,
        engine=config.engine,
        model=config.model,
        gpu_name=instance.gpus[0].name if instance.gpus else "unknown",
        gpu_count=len(instance.gpus) if instance.gpus else 1,
        workload=workload,
        host=instance.host_metadata,
        gpus=instance.gpus,
    )

    env_vars = config.build_container_env()

    # Build docker run command
    cmd_parts = ["docker", "run", "--rm", "--gpus", "all"]
    for k, v in env_vars.items():
        cmd_parts.extend(["-e", f"{k}={v}"])
    cmd_parts.append(config.bench_image)

    console.print(f"[bold]Running benchmark container locally...[/bold]")
    console.print(f"[dim]Image: {config.bench_image}[/dim]")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=config.timeout_s,
        )
        raw_output = proc.stdout + proc.stderr
        elapsed = time.time() - t0

        if proc.returncode != 0:
            result.errors.append(f"Container exited with code {proc.returncode}")
            console.print(f"[red]Container exited with code {proc.returncode}[/red]")
            # Still try to parse whatever output we got

    except subprocess.TimeoutExpired:
        result.errors.append(f"Container timed out after {config.timeout_s}s")
        console.print(f"[red]Container timed out after {config.timeout_s}s[/red]")
        return result

    # Parse the combined output
    result.raw_engine_output = raw_output
    engine_output, gpu_csv = split_container_output(raw_output)

    # Parse engine timings -- the container runs multiple iterations, each
    # producing its own timing output.  We extract per-iteration results.
    iteration_outputs = _split_iterations(engine_output, workload)
    gpu_snapshots = parse_nvidia_smi_output(gpu_csv) if gpu_csv else []

    collector = MetricsCollector()
    total_iters = workload.warmup_iterations + workload.iterations

    for i, iter_output in enumerate(iteration_outputs):
        engine_timings = _parse_engine_output(config.engine, iter_output)
        # Distribute GPU snapshots across iterations (they are aggregated)
        collector.record_iteration(i, engine_timings, gpu_snapshots or None)

    # If we got no per-iteration splits, try to parse the whole engine output
    if not iteration_outputs:
        engine_timings = _parse_engine_output(config.engine, engine_output)
        collector.record_iteration(0, engine_timings, gpu_snapshots or None)

    result.iterations = collector.iteration_metrics
    result.aggregate = collector.aggregate(warmup_iterations=workload.warmup_iterations)

    console.print(f"[green]Benchmark completed in {elapsed:.1f}s[/green]")
    return result


# ---------------------------------------------------------------------------
# Cloud benchmark (Vast.ai / RunPod log-based)
# ---------------------------------------------------------------------------


def run_cloud_benchmark(
    config: JobConfig,
    instance: ProvisionedInstance,
    workload: WorkloadConfig,
    logs: str,
) -> BenchmarkResult:
    """Parse benchmark results from cloud container logs.

    For cloud providers, the container is launched with the bench image and
    env vars directly.  The provider polls until the container exits and
    returns the full log output.  This function parses that output.
    """
    run_id = f"cloud-{uuid.uuid4().hex[:12]}"
    result = BenchmarkResult(
        run_id=run_id,
        provider=config.provider,
        engine=config.engine,
        model=config.model,
        gpu_name=instance.gpus[0].name if instance.gpus else (
            instance.extra.get("gpu_name", "unknown")
        ),
        gpu_count=instance.extra.get("gpu_count", config.provider_config.gpu_count),
        workload=workload,
        host=instance.host_metadata,
        gpus=instance.gpus,
    )

    result.raw_engine_output = logs
    engine_output, gpu_csv = split_container_output(logs)

    iteration_outputs = _split_iterations(engine_output, workload)
    gpu_snapshots = parse_nvidia_smi_output(gpu_csv) if gpu_csv else []

    collector = MetricsCollector()

    for i, iter_output in enumerate(iteration_outputs):
        engine_timings = _parse_engine_output(config.engine, iter_output)
        collector.record_iteration(i, engine_timings, gpu_snapshots or None)

    if not iteration_outputs:
        engine_timings = _parse_engine_output(config.engine, engine_output)
        collector.record_iteration(0, engine_timings, gpu_snapshots or None)

    result.iterations = collector.iteration_metrics
    result.aggregate = collector.aggregate(warmup_iterations=workload.warmup_iterations)

    return result


# ---------------------------------------------------------------------------
# Iteration splitting
# ---------------------------------------------------------------------------


def _split_iterations(engine_output: str, workload: WorkloadConfig) -> list[str]:
    """Split engine output into per-iteration chunks.

    The bench container prints markers like:
      ``--- Warmup iteration 1/1 ---``
      ``--- Iteration 1/5 ---``

    We split on those markers and return one string per iteration.
    """
    # Match both warmup and real iteration markers from entrypoint.sh
    pattern = r"---\s+(?:Warmup iteration|Iteration)\s+\d+/\d+\s+---"
    parts = re.split(pattern, engine_output)

    # The first element is anything before the first marker (container boot logs)
    # Skip it; the rest are iteration outputs
    if len(parts) > 1:
        return [p.strip() for p in parts[1:] if p.strip()]

    # No markers found -- return the whole thing as a single "iteration"
    return []


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    config: JobConfig,
    instance: ProvisionedInstance,
    workload: WorkloadConfig,
) -> BenchmarkResult:
    """Run the benchmark, dispatching to local or cloud as appropriate.

    For local: runs docker directly and captures output.
    For cloud: the provider has already launched the container and collected
    logs -- they are stored in instance.extra["logs"].
    """
    if instance.is_local:
        return run_local_benchmark(config, instance, workload)
    else:
        logs = instance.extra.get("logs", "")
        if not logs:
            raise RuntimeError(
                "Cloud provider did not return container logs. "
                "The provider must set instance.extra['logs'] after the run completes."
            )
        return run_cloud_benchmark(config, instance, workload, logs)
