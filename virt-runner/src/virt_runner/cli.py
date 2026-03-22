"""CLI entrypoint for virt-runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from virt_runner.benchmark import run_benchmark
from virt_runner.config import JobConfig
from virt_runner.host_info import collect_host_metadata, detect_local_gpus
from virt_runner.models import InferenceEngine, Provider
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


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


@app.command()
def run(
    model: Annotated[str, typer.Option("--model", "-m", help="Model identifier, e.g. 'llama-3-8b-q4'")],
    provider: Annotated[Provider, typer.Option("--provider", "-p", help="Provider: local, vast, runpod")] = Provider.LOCAL,
    engine: Annotated[InferenceEngine, typer.Option("--engine", "-e", help="Inference engine: llama.cpp, vllm")] = InferenceEngine.LLAMA_CPP,
    gpu_type: Annotated[str, typer.Option("--gpu-type", help="GPU type for cloud providers")] = "",
    gpu_count: Annotated[int, typer.Option("--gpu-count", help="Number of GPUs")] = 1,
    workload_file: Annotated[Optional[Path], typer.Option("--workload", "-w", help="Path to workload JSON")] = None,
    bench_image: Annotated[str, typer.Option("--image", help="Docker benchmark image")] = "gpuscale-bench:latest",
    ssh_key: Annotated[str, typer.Option("--ssh-key", help="Path to SSH private key for cloud")] = "",
    output_file: Annotated[Optional[Path], typer.Option("--output", "-o", help="Write results JSON to file")] = None,
    timeout: Annotated[int, typer.Option("--timeout", help="Benchmark timeout in seconds")] = 1800,
) -> None:
    """Run a GPU benchmark job."""
    config = JobConfig(
        provider=provider,
        engine=engine,
        model=model,
        bench_image=bench_image,
        ssh_key_path=ssh_key,
        timeout_s=timeout,
    )
    config.provider_config.gpu_type = gpu_type
    config.provider_config.gpu_count = gpu_count

    workload = config.load_workload(workload_file)

    prov = _get_provider(config)
    instance: ProvisionedInstance | None = None

    try:
        console.print(f"\n[bold]Provider:[/bold] {prov.get_name()}")
        console.print(f"[bold]Engine:[/bold]   {engine.value}")
        console.print(f"[bold]Model:[/bold]    {model}")
        console.print(f"[bold]Image:[/bold]    {bench_image}\n")

        # Provision
        instance = prov.provision()
        if instance.gpus:
            console.print(f"[green]Detected {len(instance.gpus)} GPU(s):[/green]")
            for gpu in instance.gpus:
                console.print(f"  [{gpu.index}] {gpu.name} ({gpu.memory_total_mib} MiB)")
        elif not instance.is_local:
            console.print("[yellow]GPU info will be collected during benchmark.[/yellow]")
        else:
            console.print("[red]No GPUs detected on local machine.[/red]")
            raise typer.Exit(1)

        # Wait for instance
        if not prov.wait_ready(instance):
            console.print("[red]Instance failed to become ready. Aborting.[/red]")
            raise typer.Exit(1)

        # Run benchmark
        result = run_benchmark(config, instance, workload)

        # Output results
        result_json = result.model_dump_json(indent=2)
        if output_file:
            output_file.write_text(result_json)
            console.print(f"\n[green]Results written to {output_file}[/green]")
        else:
            console.print("\n[bold]Results:[/bold]")
            console.print(result_json)

        # Summary table
        agg = result.aggregate
        table = Table(title="Benchmark Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Tokens/sec (mean)", f"{agg.tokens_per_sec_mean:.2f}")
        table.add_row("Tokens/sec (std)", f"{agg.tokens_per_sec_std:.2f}")
        table.add_row("TTFT (mean)", f"{agg.ttft_mean_ms:.2f} ms")
        table.add_row("Peak VRAM", f"{agg.peak_vram_mib:.0f} MiB")
        table.add_row("Avg Power", f"{agg.power_draw_avg_w:.1f} W")
        table.add_row("Peak Power", f"{agg.power_draw_peak_w:.1f} W")
        table.add_row("GPU Util (mean)", f"{agg.gpu_utilization_pct_mean:.1f}%")
        table.add_row("GPU Temp (max)", f"{agg.gpu_temperature_c_max:.1f} C")
        table.add_row("Total Wall Time", f"{agg.wall_time_total_s:.2f} s")
        console.print(table)

        if result.errors:
            console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
            for err in result.errors:
                console.print(f"  - {err}")

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


if __name__ == "__main__":
    app()
