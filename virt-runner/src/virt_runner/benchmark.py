"""Orchestrate benchmark runs: pull container, execute workload, collect metrics."""

from __future__ import annotations

import json
import subprocess
import time
import uuid

import paramiko
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from virt_runner.config import JobConfig
from virt_runner.metrics import (
    MetricsCollector,
    parse_llamacpp_output,
    parse_nvidia_smi_output,
    parse_vllm_output,
    query_nvidia_smi,
)
from virt_runner.models import BenchmarkResult, InferenceEngine, WorkloadConfig
from virt_runner.providers.base import ProvisionedInstance

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------


def _ssh_connect(instance: ProvisionedInstance) -> paramiko.SSHClient:
    """Open an SSH connection to a provisioned instance."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict = {
        "hostname": instance.host,
        "port": instance.port,
        "username": instance.ssh_user,
        "timeout": 30,
    }
    if instance.ssh_key_path:
        kwargs["key_filename"] = instance.ssh_key_path
    client.connect(**kwargs)
    return client


def _ssh_exec(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> str:
    """Execute a command over SSH and return combined stdout+stderr."""
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out + err


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _build_docker_run_cmd(
    config: JobConfig,
    workload: WorkloadConfig,
    s3_env: dict[str, str],
) -> str:
    """Build the 'docker run' command string for the benchmark container."""
    env_flags: list[str] = []
    for k, v in s3_env.items():
        env_flags.append(f"-e {k}={v}")

    workload_json = json.dumps(workload.model_dump(), separators=(",", ":"))
    env_flags.append(f"-e WORKLOAD_CONFIG='{workload_json}'")
    env_flags.append(f"-e MODEL={config.model}")
    env_flags.append(f"-e ENGINE={config.engine.value}")

    env_str = " ".join(env_flags)
    return (
        f"docker run --rm --gpus all {env_str} "
        f"{config.bench_image}"
    )


def _get_s3_env(config: JobConfig) -> dict[str, str]:
    """Build S3-related env vars for the container."""
    s3 = config.s3
    env: dict[str, str] = {}
    if s3.endpoint:
        env["S3_ENDPOINT"] = s3.endpoint
    if s3.access_key:
        env["AWS_ACCESS_KEY_ID"] = s3.access_key
    if s3.secret_key:
        env["AWS_SECRET_ACCESS_KEY"] = s3.secret_key
    if s3.bucket:
        env["S3_BUCKET"] = s3.bucket
    if s3.model_key:
        env["S3_MODEL_KEY"] = s3.model_key
    return env


# ---------------------------------------------------------------------------
# Engine output parsing dispatch
# ---------------------------------------------------------------------------


def _parse_engine_output(engine: InferenceEngine, output: str):
    """Route engine output to the appropriate parser."""
    if engine == InferenceEngine.LLAMA_CPP:
        return parse_llamacpp_output(output)
    elif engine == InferenceEngine.VLLM:
        return parse_vllm_output(output)
    else:
        return parse_llamacpp_output(output)  # fallback


# ---------------------------------------------------------------------------
# Remote benchmark (cloud providers)
# ---------------------------------------------------------------------------


def run_remote_benchmark(
    config: JobConfig,
    instance: ProvisionedInstance,
    workload: WorkloadConfig,
) -> BenchmarkResult:
    """Run a benchmark on a remote instance via SSH."""
    run_id = f"remote-{uuid.uuid4().hex[:12]}"
    result = BenchmarkResult(
        run_id=run_id,
        provider=config.provider,
        engine=config.engine,
        model=config.model,
        gpu_name=instance.gpus[0].name if instance.gpus else "",
        gpu_count=len(instance.gpus) if instance.gpus else config.provider_config.gpu_count,
        workload=workload,
        host=instance.host_metadata,
        gpus=instance.gpus,
    )

    ssh = _ssh_connect(instance)
    try:
        # Pull the benchmark container
        console.print(f"[bold]Pulling benchmark image {config.bench_image}...[/bold]")
        pull_output = _ssh_exec(ssh, f"docker pull {config.bench_image}", timeout=600)
        console.print(f"[dim]{pull_output[:200]}[/dim]")

        s3_env = _get_s3_env(config)
        docker_cmd = _build_docker_run_cmd(config, workload, s3_env)
        collector = MetricsCollector()

        total_iterations = workload.warmup_iterations + workload.iterations
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmark...", total=total_iterations)

            for i in range(total_iterations):
                is_warmup = i < workload.warmup_iterations
                label = f"warmup {i + 1}" if is_warmup else f"iteration {i + 1 - workload.warmup_iterations}"
                progress.update(task, description=f"Running {label}...")

                # Collect GPU snapshot before run
                smi_raw = _ssh_exec(
                    ssh,
                    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,"
                    "power.draw,temperature.gpu --format=csv,noheader,nounits",
                    timeout=15,
                )

                # Run the benchmark container
                output = _ssh_exec(ssh, docker_cmd, timeout=config.timeout_s)

                # Parse results
                engine_timings = _parse_engine_output(config.engine, output)
                gpu_snapshots = parse_nvidia_smi_output(smi_raw)
                collector.record_iteration(i, engine_timings, gpu_snapshots or None)

                progress.advance(task)

        result.iterations = collector.iteration_metrics
        result.aggregate = collector.aggregate(warmup_iterations=workload.warmup_iterations)
        result.raw_engine_output = output  # last iteration output
    except Exception as exc:
        result.errors.append(str(exc))
        console.print(f"[red]Benchmark error: {exc}[/red]")
    finally:
        ssh.close()

    return result


# ---------------------------------------------------------------------------
# Local benchmark
# ---------------------------------------------------------------------------


def run_local_benchmark(
    config: JobConfig,
    instance: ProvisionedInstance,
    workload: WorkloadConfig,
) -> BenchmarkResult:
    """Run a benchmark on the local machine using Docker."""
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

    s3_env = _get_s3_env(config)
    docker_cmd = _build_docker_run_cmd(config, workload, s3_env)
    collector = MetricsCollector()

    total_iterations = workload.warmup_iterations + workload.iterations
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmark...", total=total_iterations)

        for i in range(total_iterations):
            is_warmup = i < workload.warmup_iterations
            label = f"warmup {i + 1}" if is_warmup else f"iteration {i + 1 - workload.warmup_iterations}"
            progress.update(task, description=f"Running {label}...")

            # Collect GPU snapshot
            gpu_snapshots = query_nvidia_smi()

            # Run the benchmark container locally
            try:
                proc = subprocess.run(
                    docker_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_s,
                )
                output = proc.stdout + proc.stderr
            except subprocess.TimeoutExpired:
                result.errors.append(f"Iteration {i} timed out after {config.timeout_s}s")
                output = ""

            # Parse results
            engine_timings = _parse_engine_output(config.engine, output)
            collector.record_iteration(i, engine_timings, gpu_snapshots or None)

            progress.advance(task)

    result.iterations = collector.iteration_metrics
    result.aggregate = collector.aggregate(warmup_iterations=workload.warmup_iterations)
    result.raw_engine_output = output  # last iteration output

    return result


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    config: JobConfig,
    instance: ProvisionedInstance,
    workload: WorkloadConfig,
) -> BenchmarkResult:
    """Run the benchmark, dispatching to local or remote as appropriate."""
    if instance.is_local:
        return run_local_benchmark(config, instance, workload)
    else:
        return run_remote_benchmark(config, instance, workload)
