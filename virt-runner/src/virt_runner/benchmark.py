"""Orchestrate benchmark runs: run container, parse combined output, collect metrics."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

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
# Local no-Docker benchmark (runs bench-container scripts directly on host)
# ---------------------------------------------------------------------------


def _find_bench_container_dir() -> Path | None:
    """Locate the bench-container directory by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "bench-container"
        if (candidate / "scripts").is_dir():
            return candidate
    return None


def _build_engine_cmd(
    engine: InferenceEngine,
    bench_dir: Path,
    prompt: str,
    gen_params,
) -> list[str]:
    """Construct the subprocess argv for a single engine iteration."""
    scripts = bench_dir / "scripts"
    max_tokens = str(gen_params.max_tokens)
    temperature = str(gen_params.temperature)
    top_p = str(gen_params.top_p)

    if engine == InferenceEngine.LLAMA_CPP:
        return [
            "bash",
            str(scripts / "run_llama_cpp.sh"),
            prompt,
            max_tokens,
            temperature,
            top_p,
        ]
    if engine == InferenceEngine.VLLM:
        return [
            "uv",
            "run",
            "--project",
            str(bench_dir),
            "python",
            str(scripts / "run_vllm.py"),
            prompt,
            max_tokens,
            temperature,
            top_p,
        ]
    raise ValueError(f"Unsupported engine for no-docker mode: {engine}")


def run_local_benchmark_no_docker(
    config: JobConfig,
    instance: ProvisionedInstance,
    workload: WorkloadConfig,
) -> BenchmarkResult:
    """Run the benchmark directly on the host without Docker.

    Mirrors bench-container/entrypoint.sh: pulls the model via pull_model.py,
    starts collect_metrics.sh (no-ops if nvidia-smi is missing), runs the
    engine scripts for each warmup + real iteration, then parses the combined
    output through the same pipeline as the Docker path.
    """
    run_id = f"local-nodocker-{uuid.uuid4().hex[:12]}"
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

    bench_dir = _find_bench_container_dir()
    if bench_dir is None:
        result.errors.append(
            "Could not locate bench-container/ directory relative to virt-runner."
        )
        console.print("[red]bench-container directory not found.[/red]")
        return result
    scripts_dir = bench_dir / "scripts"

    # Model dir: GPUSCALE_MODEL_DIR override, else ~/.cache/gpuscale/models/<model>/<format>
    fmt = config.model_format or "full"
    model_dir_env = os.environ.get("GPUSCALE_MODEL_DIR")
    if model_dir_env:
        model_dir = Path(model_dir_env)
    else:
        model_dir = (
            Path.home() / ".cache" / "gpuscale" / "models" / config.model / fmt
        )
    model_dir.mkdir(parents=True, exist_ok=True)

    # Env for subprocesses: inherit + bench container env + MODEL_DIR override
    env = os.environ.copy()
    env.update(config.build_container_env())
    env["MODEL_DIR"] = str(model_dir)

    # ---- Step 1: Pull model ----
    console.print(f"[bold]Pulling model into {model_dir}...[/bold]")
    try:
        pull_proc = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(bench_dir),
                "python",
                str(scripts_dir / "pull_model.py"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_s,
        )
        if pull_proc.returncode != 0:
            err_tail = (pull_proc.stderr or "")[-800:]
            result.errors.append(f"Model pull failed: {err_tail.strip()}")
            console.print(f"[red]Model pull failed[/red]\n{err_tail}")
            return result
    except FileNotFoundError:
        result.errors.append("uv is not on PATH; required for no-docker mode.")
        console.print("[red]`uv` not found on PATH.[/red]")
        return result
    except subprocess.TimeoutExpired:
        result.errors.append(f"Model pull timed out after {config.timeout_s}s")
        return result

    # ---- Step 2: Detect format on disk and route engine ----
    has_gguf = any(model_dir.glob("*.gguf"))
    has_pth = any(model_dir.glob("*.pth"))

    engine = config.engine
    if has_pth:
        result.errors.append(
            "Meta .pth format detected; meta-native engine is not supported in no-docker mode."
        )
        console.print(
            "[red]Meta .pth detected — no-docker mode only supports llama.cpp and vllm.[/red]"
        )
        return result
    if has_gguf and engine != InferenceEngine.LLAMA_CPP:
        console.print(
            "[yellow]GGUF detected — switching engine to llama.cpp.[/yellow]"
        )
        engine = InferenceEngine.LLAMA_CPP
        result.engine = engine

    # ---- Step 3: Start GPU metrics collection ----
    metrics_csv = Path("/tmp/gpu_metrics.csv")
    try:
        metrics_csv.unlink()
    except FileNotFoundError:
        pass

    metrics_available = shutil.which("nvidia-smi") is not None
    if metrics_available:
        subprocess.run(
            ["bash", str(scripts_dir / "collect_metrics.sh"), "start"],
            env=env,
            check=False,
        )
    else:
        console.print(
            "[yellow]nvidia-smi not found; GPU metrics will be empty.[/yellow]"
        )

    # ---- Step 4: Run iterations ----
    engine_log_parts: list[str] = []
    total_iters = workload.warmup_iterations + workload.iterations
    num_prompts = len(workload.prompts)
    successes = 0
    failures = 0

    console.print(
        f"[bold]Running {total_iters} iterations x {num_prompts} prompts "
        f"({engine.value})...[/bold]"
    )

    t0 = time.time()
    for i in range(1, total_iters + 1):
        if i <= workload.warmup_iterations:
            label = f"Warmup iteration {i}/{workload.warmup_iterations}"
        else:
            real_iter = i - workload.warmup_iterations
            label = f"Iteration {real_iter}/{workload.iterations}"

        for p_idx, prompt_msg in enumerate(workload.prompts):
            marker = f"--- {label}, prompt {p_idx + 1}/{num_prompts} ---"
            console.print(f"[dim]{marker}[/dim]")
            engine_log_parts.append(marker)

            try:
                cmd = _build_engine_cmd(
                    engine, bench_dir, prompt_msg.content, workload.generation_params
                )
            except ValueError as exc:
                engine_log_parts.append(f"[error] {exc}")
                failures += 1
                continue

            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_s,
                )
                combined = (proc.stdout or "") + (proc.stderr or "")
                engine_log_parts.append(combined)
                if proc.returncode == 0:
                    successes += 1
                else:
                    failures += 1
                    console.print(
                        f"[yellow]WARNING: {engine.value} exited {proc.returncode}[/yellow]"
                    )
            except FileNotFoundError as exc:
                engine_log_parts.append(f"[error] command not found: {exc}")
                failures += 1
                console.print(f"[red]Command not found: {exc}[/red]")
            except subprocess.TimeoutExpired:
                engine_log_parts.append(
                    f"[error] iteration timed out after {config.timeout_s}s"
                )
                failures += 1

    elapsed = time.time() - t0

    # ---- Step 5: Stop metrics ----
    if metrics_available:
        subprocess.run(
            ["bash", str(scripts_dir / "collect_metrics.sh"), "stop"],
            env=env,
            check=False,
        )

    # ---- Step 6: Assemble marker-wrapped output and parse ----
    engine_log = "\n".join(engine_log_parts)
    gpu_csv = ""
    if metrics_csv.exists():
        try:
            gpu_csv = metrics_csv.read_text()
        except OSError:
            gpu_csv = ""

    raw_output = (
        f"{ENGINE_OUTPUT_START}\n{engine_log}\n{ENGINE_OUTPUT_END}\n"
        f"{GPU_METRICS_START}\n{gpu_csv}\n{GPU_METRICS_END}\n"
    )
    result.raw_engine_output = raw_output

    engine_output, gpu_csv_parsed = split_container_output(raw_output)
    iteration_outputs = _split_iterations(engine_output, workload)
    gpu_snapshots = parse_nvidia_smi_output(gpu_csv_parsed) if gpu_csv_parsed else []

    collector = MetricsCollector()
    for i, iter_output in enumerate(iteration_outputs):
        engine_timings = _parse_engine_output(engine, iter_output)
        collector.record_iteration(i, engine_timings, gpu_snapshots or None)

    if not iteration_outputs:
        engine_timings = _parse_engine_output(engine, engine_output)
        collector.record_iteration(0, engine_timings, gpu_snapshots or None)

    result.iterations = collector.iteration_metrics
    result.aggregate = collector.aggregate(
        warmup_iterations=workload.warmup_iterations
    )

    if failures:
        result.errors.append(
            f"{failures}/{successes + failures} iterations failed"
        )

    console.print(
        f"[green]Benchmark completed in {elapsed:.1f}s "
        f"({successes} succeeded, {failures} failed)[/green]"
    )
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
    # Format: "--- Iteration 1/3, prompt 1/3 ---" or "--- Warmup iteration 1/1, prompt 1/3 ---"
    pattern = r"---\s+(?:Warmup iteration|Iteration)\s+\d+/\d+.*?---"
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
        if config.no_docker:
            return run_local_benchmark_no_docker(config, instance, workload)
        return run_local_benchmark(config, instance, workload)
    else:
        logs = instance.extra.get("logs", "")
        if not logs:
            raise RuntimeError(
                "Cloud provider did not return container logs. "
                "The provider must set instance.extra['logs'] after the run completes."
            )
        return run_cloud_benchmark(config, instance, workload, logs)
