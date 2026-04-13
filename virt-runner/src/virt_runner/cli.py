"""CLI entrypoint for virt-runner."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv

# Walk up from this file to find the monorepo root .env
_env_path = Path(__file__).resolve()
for _parent in _env_path.parents:
    _candidate = _parent / ".env"
    if _candidate.exists():
        load_dotenv(_candidate)
        break

import typer
from rich.console import Console
from rich.table import Table

from virt_runner.benchmark import run_benchmark
from virt_runner.config import JobConfig
from virt_runner.host_info import collect_host_metadata, detect_local_gpus
from virt_runner.models import BenchmarkResult, InferenceEngine, Provider
from virt_runner.providers.base import ProvisionedInstance
from virt_runner.providers.local import LocalProvider
from virt_runner.providers.runpod import RunPodProvider
from virt_runner.providers.vast import VastProvider

app = typer.Typer(
    name="virt-runner",
    help="GPU benchmarking orchestrator for cloud and local instances.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _get_provider(config: JobConfig):
    """Instantiate the appropriate provider based on config."""
    match config.provider:
        case Provider.LOCAL:
            return LocalProvider(config)
        case Provider.VAST:
            return VastProvider(config)
        case Provider.RUNPOD:
            return RunPodProvider(config)
        case _:
            raise typer.BadParameter(f"Unknown provider: {config.provider}")


def _build_dbops_payload(result: BenchmarkResult, config: JobConfig) -> dict:
    """Convert a BenchmarkResult into a dict matching dbops BenchmarkResultCreate schema."""
    agg = result.aggregate

    # Map provider names to what the DB constraint expects
    is_community = config.provider_config.extra.get("community", False)
    provider_map = {
        "local": "local",
        "vast": "vast.ai (community)" if is_community else "vast.ai",
        "runpod": "runpod",
    }
    db_provider = provider_map.get(result.provider.value, result.provider.value)

    # Derive quantization label
    quant = "FP16"
    if config.gguf_quant:
        quant = config.gguf_quant
    elif config.model_format:
        fmt = config.model_format.lower()
        if fmt == "gptq":
            quant = "GPTQ-4bit"
        elif fmt == "gguf":
            quant = config.gguf_quant or "GGUF"
        elif fmt == "full":
            quant = "FP16"

    # GPU VRAM in GB (from GPUInfo or offer data)
    gpu_vram_gb = 0.0
    if result.gpus:
        gpu_vram_gb = max(g.memory_total_mib for g in result.gpus) / 1024.0
    if gpu_vram_gb == 0 and result.extra.get("gpu_vram_gb"):
        gpu_vram_gb = float(result.extra["gpu_vram_gb"])
    if gpu_vram_gb == 0:
        gpu_vram_gb = 24.0  # fallback for common GPUs

    payload = {
        "gpu_name": result.gpu_name or "unknown",
        "gpu_vram_gb": round(gpu_vram_gb, 1),
        "gpu_count": result.gpu_count,
        "provider": db_provider,
        "engine": result.engine.value,
        "model_name": result.model,
        "quantization": quant,
        "workload_version": result.workload.workload_version if result.workload else "1.0",
        "workload_config": result.workload.model_dump() if result.workload else None,
        "tokens_per_sec": max(agg.tokens_per_sec_mean, 0.01),  # DB requires > 0
        "time_to_first_token_ms": max(agg.ttft_mean_ms, 0.0),
        "prompt_eval_tokens_per_sec": agg.prompt_eval_rate_mean or None,
        "peak_vram_mb": agg.peak_vram_mib or None,
        "avg_power_draw_w": agg.power_draw_avg_w or None,
        "peak_power_draw_w": agg.power_draw_peak_w or None,
        "avg_gpu_util_pct": agg.gpu_utilization_pct_mean or None,
        "avg_gpu_temp_c": agg.gpu_temperature_c_max or None,
        "total_wall_time_s": agg.wall_time_total_s or None,
        "engine_version": None,
        "host_os": f"{result.host.os} {result.host.distro}".strip() or None,
        "host_kernel": result.host.kernel_version or None,
        "host_driver_version": result.host.gpu_driver_version or None,
        "container_image": config.bench_image,
        "container_driver_version": None,
        "raw_output": {
            "run_id": result.run_id,
            "iterations": [m.model_dump() for m in result.iterations],
            "errors": result.errors,
        },
    }
    return payload


def _submit_to_db(payload: dict) -> None:
    """Submit benchmark result to the database via dbops."""
    try:
        from dbops.db import get_session, insert_result
        from dbops.models import BenchmarkResultCreate
    except ImportError:
        console.print(
            "[red]Cannot import dbops. Install it or ensure it is on the Python path.[/red]\n"
            "[dim]Try: pip install -e ../dbops  or  uv pip install -e ../dbops[/dim]"
        )
        raise typer.Exit(1)

    # Validate through Pydantic
    try:
        validated = BenchmarkResultCreate(**payload)
    except Exception as exc:
        console.print(f"[red]Validation error: {exc}[/red]")
        console.print(f"[dim]Payload: {json.dumps(payload, indent=2, default=str)}[/dim]")
        raise typer.Exit(1)

    # Insert into DB
    try:
        with get_session() as session:
            orm_obj = validated.to_orm()
            inserted = insert_result(session, orm_obj)
            console.print(
                f"[green]Result submitted to database. ID: {inserted.id}[/green]"
            )
    except Exception as exc:
        console.print(f"[red]Database error: {exc}[/red]")
        raise typer.Exit(1)


def _print_summary(result: BenchmarkResult) -> None:
    """Print a rich summary table of the benchmark result."""
    agg = result.aggregate
    table = Table(title="Benchmark Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Provider", result.provider.value)
    table.add_row("Engine", result.engine.value)
    table.add_row("Model", result.model)
    table.add_row("GPU", f"{result.gpu_name} x{result.gpu_count}")
    table.add_row("", "")
    table.add_row("Tokens/sec (mean)", f"{agg.tokens_per_sec_mean:.2f}")
    table.add_row("Tokens/sec (std)", f"{agg.tokens_per_sec_std:.2f}")
    table.add_row("TTFT (mean)", f"{agg.ttft_mean_ms:.2f} ms")
    table.add_row("Prompt eval rate", f"{agg.prompt_eval_rate_mean:.2f} tok/s")
    table.add_row("", "")
    table.add_row("Peak VRAM", f"{agg.peak_vram_mib:.0f} MiB")
    table.add_row("Avg Power", f"{agg.power_draw_avg_w:.1f} W")
    table.add_row("Peak Power", f"{agg.power_draw_peak_w:.1f} W")
    table.add_row("GPU Util (mean)", f"{agg.gpu_utilization_pct_mean:.1f}%")
    table.add_row("GPU Temp (max)", f"{agg.gpu_temperature_c_max:.1f} C")
    table.add_row("Total Wall Time", f"{agg.wall_time_total_s:.2f} s")
    console.print(table)


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


@app.command()
def run(
    model: Annotated[str, typer.Option(
        "--model", "-m", help="Model identifier, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
    )],
    engine: Annotated[InferenceEngine, typer.Option(
        "--engine", "-e", help="Inference engine: llama.cpp, vllm"
    )] = InferenceEngine.LLAMA_CPP,
    provider: Annotated[Provider, typer.Option(
        "--provider", "-p", help="Provider: local, vast, runpod"
    )] = Provider.LOCAL,
    gpu_type: Annotated[str, typer.Option(
        "--gpu", help="GPU type for cloud providers, e.g. 'RTX_4090'"
    )] = "",
    gpu_count: Annotated[int, typer.Option(
        "--gpu-count", help="Number of GPUs"
    )] = 1,
    model_format: Annotated[str, typer.Option(
        "--model-format", help="Model format: full, gguf, gptq"
    )] = "",
    gguf_quant: Annotated[str, typer.Option(
        "--gguf-quant", help="GGUF quantization, e.g. Q4_K_M"
    )] = "",
    workload_file: Annotated[Optional[Path], typer.Option(
        "--workload", "-w", help="Path to workload JSON"
    )] = None,
    bench_image: Annotated[str, typer.Option(
        "--image", help="Docker benchmark image"
    )] = "ghcr.io/cmink2040/gpuscale-bench:latest",
    output_file: Annotated[Optional[Path], typer.Option(
        "--output", "-o", help="Write results JSON to file"
    )] = None,
    timeout: Annotated[int, typer.Option(
        "--timeout", help="Benchmark timeout in seconds"
    )] = 1800,
    submit: Annotated[bool, typer.Option(
        "--submit", help="Submit results to the database via dbops"
    )] = False,
    community: Annotated[bool, typer.Option(
        "--community", help="Use community cloud instead of datacenter (Vast.ai)"
    )] = False,
    no_docker: Annotated[bool, typer.Option(
        "--no-docker",
        help="Local only: run bench-container scripts directly on the host instead of via Docker.",
    )] = False,
) -> None:
    """Run a GPU benchmark job."""
    config = JobConfig(
        provider=provider,
        engine=engine,
        model=model,
        model_format=model_format,
        gguf_quant=gguf_quant,
        bench_image=bench_image,
        timeout_s=timeout,
        no_docker=no_docker,
    )
    if no_docker and provider != Provider.LOCAL:
        raise typer.BadParameter("--no-docker is only supported with --provider local")
    config.provider_config.gpu_type = gpu_type
    config.provider_config.gpu_count = gpu_count
    config.provider_config.extra["community"] = community

    workload = config.load_workload(workload_file)

    prov = _get_provider(config)
    instance: ProvisionedInstance | None = None

    try:
        console.print()
        console.print(f"[bold]Provider:[/bold]     {prov.get_name()}")
        console.print(f"[bold]Engine:[/bold]       {engine.value}")
        console.print(f"[bold]Model:[/bold]        {model}")
        if model_format:
            console.print(f"[bold]Format:[/bold]       {model_format}")
        if gguf_quant:
            console.print(f"[bold]Quantization:[/bold] {gguf_quant}")
        console.print(f"[bold]Image:[/bold]        {bench_image}")
        console.print()

        # Provision
        instance = prov.provision()
        if instance.gpus:
            console.print(f"[green]GPU(s): {len(instance.gpus)}[/green]")
            for gpu in instance.gpus:
                vram = f" ({gpu.memory_total_mib} MiB)" if gpu.memory_total_mib else ""
                console.print(f"  [{gpu.index}] {gpu.name}{vram}")
        elif provider == Provider.LOCAL:
            console.print(
                "[yellow]No GPUs detected locally. The container needs --gpus all "
                "and nvidia-container-toolkit installed.[/yellow]"
            )

        # Wait for instance / container to finish
        if not prov.wait_ready(instance, timeout_s=timeout):
            console.print("[red]Instance/container failed or timed out. Aborting.[/red]")
            raise typer.Exit(1)

        # Run benchmark (local) or parse logs (cloud)
        result = run_benchmark(config, instance, workload)

        # Always save the result JSON
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        default_output = Path(f"bench_result_{timestamp}.json")
        out_path = output_file or default_output

        result_dict = result.model_dump(mode="json")
        result_json = json.dumps(result_dict, indent=2, default=str)
        out_path.write_text(result_json)
        console.print(f"\n[green]Results written to {out_path}[/green]")

        # Print summary
        console.print()
        _print_summary(result)

        # Print errors if any
        if result.errors:
            console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
            for err in result.errors:
                console.print(f"  - {err}")

        # Submit to DB if requested
        if submit:
            console.print("\n[bold]Submitting to database...[/bold]")
            payload = _build_dbops_payload(result, config)
            _submit_to_db(payload)

    finally:
        if instance is not None:
            prov.teardown(instance)


# ---------------------------------------------------------------------------
# list-gpus command
# ---------------------------------------------------------------------------


@app.command(name="list-gpus")
def list_gpus() -> None:
    """Detect and display local GPUs."""
    gpus = detect_local_gpus()
    meta = collect_host_metadata()

    if not gpus:
        console.print("[yellow]No GPUs detected.[/yellow]")
        console.print(
            "Ensure nvidia-smi or rocm-smi is installed and accessible."
        )
        raise typer.Exit(1)

    console.print(f"\n[bold]Host:[/bold] {meta.hostname}")
    console.print(f"[bold]OS:[/bold]   {meta.os} {meta.distro}")
    console.print(f"[bold]Kernel:[/bold] {meta.kernel_version}")
    console.print(f"[bold]Driver:[/bold] {meta.gpu_driver_version}")
    console.print(f"[bold]CUDA:[/bold]   {meta.cuda_version}")
    console.print(f"[bold]Docker:[/bold] {meta.docker_runtime_version}\n")

    table = Table(title=f"Local GPUs ({len(gpus)} detected)")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Memory (MiB)", justify="right")
    table.add_column("UUID", style="dim")
    table.add_column("PCI Bus", style="dim")

    for gpu in gpus:
        table.add_row(
            str(gpu.index),
            gpu.name,
            str(gpu.memory_total_mib),
            gpu.uuid[:16] + "..." if len(gpu.uuid) > 16 else gpu.uuid,
            gpu.pci_bus_id,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# host-info command
# ---------------------------------------------------------------------------


@app.command(name="host-info")
def host_info(
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """Display host system metadata."""
    meta = collect_host_metadata()
    if json_output:
        console.print(meta.model_dump_json(indent=2))
    else:
        console.print(f"[bold]Hostname:[/bold] {meta.hostname}")
        console.print(f"[bold]OS:[/bold]       {meta.os}")
        console.print(f"[bold]Distro:[/bold]   {meta.distro}")
        console.print(f"[bold]Kernel:[/bold]   {meta.kernel_version}")
        console.print(f"[bold]Driver:[/bold]   {meta.gpu_driver_version}")
        console.print(f"[bold]Docker:[/bold]   {meta.docker_runtime_version}")
        console.print(f"[bold]CUDA:[/bold]     {meta.cuda_version}")


# ---------------------------------------------------------------------------
# RunPod volume management
# ---------------------------------------------------------------------------


@app.command(name="create-volume")
def create_volume(
    name: Annotated[str, typer.Option(help="Volume name")] = "gpuscale-models",
    size: Annotated[int, typer.Option(help="Size in GB")] = 50,
    region: Annotated[str, typer.Option(help="RunPod data center ID")] = "US-TX-3",
) -> None:
    """Create a RunPod network volume for persistent model storage."""
    config = JobConfig(provider=Provider.RUNPOD)
    provider = RunPodProvider(config)
    vol_id = provider.create_network_volume(name=name, size_gb=size, region=region)
    console.print(f"\nAdd this to your .env:\n  [bold]RUNPOD_VOLUME_ID={vol_id}[/bold]")


@app.command(name="sync-volume")
def sync_volume(
    volume_id: Annotated[str, typer.Option(
        "--volume-id", help="RunPod volume ID (or set RUNPOD_VOLUME_ID env var)"
    )] = "",
) -> None:
    """Sync models from Wasabi S3 to a RunPod network volume."""
    vol_id = volume_id or os.environ.get("RUNPOD_VOLUME_ID", "")
    if not vol_id:
        console.print("[red]Volume ID required. Pass --volume-id or set RUNPOD_VOLUME_ID.[/red]")
        raise typer.Exit(1)

    config = JobConfig(provider=Provider.RUNPOD)
    provider = RunPodProvider(config)
    provider.sync_s3_to_volume(vol_id)


@app.command(name="list-volumes")
def list_volumes_cmd() -> None:
    """List RunPod network volumes."""
    config = JobConfig(provider=Provider.RUNPOD)
    provider = RunPodProvider(config)
    volumes = provider.list_volumes()
    if not volumes:
        console.print("[dim]No network volumes found.[/dim]")
        return
    table = Table(title="RunPod Network Volumes")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Region", style="dim")
    for v in volumes:
        table.add_row(v["id"], v["name"], str(v.get("size", "")), v.get("dataCenterId", ""))
    console.print(table)


if __name__ == "__main__":
    app()
