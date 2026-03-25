"""Vast.ai provider: deploy bench container directly and collect logs via vastai CLI."""

from __future__ import annotations

import json
import os
import subprocess
import time

from rich.console import Console

from virt_runner.config import JobConfig
from virt_runner.models import GPUInfo
from virt_runner.providers.base import BaseProvider, ProvisionedInstance

console = Console(stderr=True)

# Timeout for individual vastai CLI calls (seconds)
_CLI_TIMEOUT = 120


def _vast_cli(
    *args: str,
    api_key: str = "",
    timeout: int = _CLI_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run a vastai CLI command, injecting the API key via env var."""
    env = os.environ.copy()
    if api_key:
        env["VAST_API_KEY"] = api_key
    cmd = ["vastai", *args]
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
    )


class VastProvider(BaseProvider):
    """Deploy the benchmark container on Vast.ai and collect results.

    Flow:
    1. ``vastai search offers`` to find a machine matching GPU requirements
    2. ``vastai create instance`` with the bench image + env vars
    3. Poll instance status until it exits (the container runs to completion)
    4. ``vastai logs`` to fetch stdout/stderr (contains engine output + GPU metrics)
    5. ``vastai destroy instance`` to clean up
    """

    def __init__(self, config: JobConfig) -> None:
        super().__init__(config)
        self.api_key = config.provider_config.api_key or os.getenv("VAST_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Vast.ai API key required. Set VAST_API_KEY env var or pass via provider_config."
            )

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def provision(self) -> ProvisionedInstance:
        """Search for a matching offer and create an instance with the bench image."""
        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "RTX_4090"
        gpu_count = pc.gpu_count
        disk_gb = pc.disk_gb
        max_dph = pc.max_dph

        # --- Search for offers ---
        console.print(f"[bold]Searching Vast.ai for {gpu_count}x {gpu_type}...[/bold]")
        query = (
            f"gpu_name={gpu_type} "
            f"num_gpus={gpu_count} "
            f"disk_space>={disk_gb} "
            f"dph<={max_dph} "
            f"inet_down>200"
        )
        search_result = _vast_cli(
            "search", "offers", query,
            "-o", "dph",
            "--raw",
            api_key=self.api_key,
        )
        if search_result.returncode != 0:
            raise RuntimeError(f"vastai search failed: {search_result.stderr}")

        try:
            offers = json.loads(search_result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Failed to parse vastai search output:\n{search_result.stdout[:500]}"
            )

        if not offers:
            raise RuntimeError(
                f"No Vast.ai offers found for {gpu_count}x {gpu_type} "
                f"with >={disk_gb}GB disk, <=${max_dph}$/hr. "
                f"Try relaxing --gpu-type or increasing max_dph."
            )

        # Pick the cheapest offer
        offer = min(offers, key=lambda o: o.get("dph_total", float("inf")))
        offer_id = offer["id"]
        gpu_name = offer.get("gpu_name", gpu_type)
        gpu_ram_gb = offer.get("gpu_ram", 0)
        num_gpus = offer.get("num_gpus", gpu_count)
        console.print(
            f"[green]Selected offer {offer_id}: "
            f"{gpu_name} x{num_gpus} "
            f"({gpu_ram_gb}GB VRAM) "
            f"@ ${offer.get('dph_total', '?')}/hr[/green]"
        )

        # --- Build env string for vastai ---
        # vastai create instance expects --env as a single string of -e KEY=VAL pairs
        env_vars = self.config.build_container_env()
        env_parts = []
        for k, v in env_vars.items():
            # Shell-escape the value
            escaped = v.replace("'", "'\\''")
            env_parts.append(f"-e {k}='{escaped}'")
        env_str = " ".join(env_parts)

        # --- Create instance ---
        image = self.config.bench_image
        console.print(f"[bold]Creating instance with image {image}...[/bold]")

        create_args = [
            "create", "instance", str(offer_id),
            "--image", image,
            "--disk", str(disk_gb),
            "--env", env_str,
            "--raw",
        ]
        create_result = _vast_cli(*create_args, api_key=self.api_key)
        if create_result.returncode != 0:
            raise RuntimeError(
                f"vastai create instance failed: {create_result.stderr}\n"
                f"stdout: {create_result.stdout}"
            )

        try:
            create_data = json.loads(create_result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Failed to parse vastai create output:\n{create_result.stdout[:500]}"
            )

        instance_id = str(create_data.get("new_contract", create_data.get("id", "")))
        if not instance_id:
            raise RuntimeError(f"No instance ID in create response: {create_data}")

        console.print(f"[green]Created instance {instance_id}[/green]")

        # Build GPU info from the offer data
        gpus = []
        for i in range(num_gpus):
            gpus.append(GPUInfo(
                index=i,
                name=gpu_name,
                memory_total_mib=int(gpu_ram_gb * 1024) if gpu_ram_gb else 0,
            ))

        return ProvisionedInstance(
            instance_id=instance_id,
            ssh_user="root",
            gpus=gpus,
            extra={
                "offer": offer,
                "create_response": create_data,
                "gpu_name": gpu_name,
                "gpu_count": num_gpus,
                "gpu_vram_gb": gpu_ram_gb,
            },
        )

    def wait_ready(self, instance: ProvisionedInstance, timeout_s: int = 1800) -> bool:
        """Poll until the Vast.ai instance finishes running the container.

        The bench container is the entrypoint -- it runs, prints output, and
        exits. We wait for the container to reach a terminal state, then
        fetch the logs.

        Returns True if we got logs successfully, False on timeout/error.
        """
        console.print(
            f"[bold]Waiting for instance {instance.instance_id} "
            f"to complete (timeout {timeout_s}s)...[/bold]"
        )
        deadline = time.time() + timeout_s
        poll_interval = 15  # seconds
        last_status = ""

        while time.time() < deadline:
            result = _vast_cli(
                "show", "instance", instance.instance_id,
                "--raw",
                api_key=self.api_key,
            )
            if result.returncode != 0:
                console.print(f"[yellow]Poll error: {result.stderr.strip()}[/yellow]")
                time.sleep(poll_interval)
                continue

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                time.sleep(poll_interval)
                continue

            status = data.get("actual_status", "unknown")
            if status != last_status:
                console.print(f"  Instance status: [cyan]{status}[/cyan]")
                last_status = status

            # The container is running -- keep waiting
            if status == "running":
                time.sleep(poll_interval)
                continue

            # Container exited or instance is in a terminal state
            if status in ("exited", "stopped", "offline"):
                console.print(f"[green]Container finished (status: {status})[/green]")
                logs = self._fetch_logs(instance.instance_id)
                if logs:
                    instance.extra["logs"] = logs
                    return True
                else:
                    console.print("[yellow]Container exited but no logs retrieved.[/yellow]")
                    return False

            # Instance is still loading / initializing
            if status in ("loading", "creating", "pulling", "starting"):
                time.sleep(poll_interval)
                continue

            # Unexpected status - keep trying
            time.sleep(poll_interval)

        console.print("[red]Timeout waiting for benchmark to complete.[/red]")
        # Try to grab whatever logs exist
        logs = self._fetch_logs(instance.instance_id)
        if logs:
            instance.extra["logs"] = logs
            return True
        return False

    def teardown(self, instance: ProvisionedInstance) -> None:
        """Destroy the Vast.ai instance."""
        if not instance.instance_id or instance.instance_id == "local":
            return
        console.print(f"[bold]Destroying instance {instance.instance_id}...[/bold]")
        result = _vast_cli(
            "destroy", "instance", instance.instance_id,
            api_key=self.api_key,
        )
        if result.returncode != 0:
            console.print(f"[red]Warning: teardown failed: {result.stderr}[/red]")
        else:
            console.print(f"[green]Instance {instance.instance_id} destroyed.[/green]")

    def get_name(self) -> str:
        return "vast.ai"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_logs(self, instance_id: str) -> str:
        """Fetch container logs from a Vast.ai instance."""
        console.print(f"[bold]Fetching logs for instance {instance_id}...[/bold]")
        result = _vast_cli(
            "logs", instance_id,
            api_key=self.api_key,
            timeout=60,
        )
        if result.returncode != 0:
            console.print(f"[red]Failed to fetch logs: {result.stderr}[/red]")
            return ""

        logs = result.stdout
        if not logs.strip():
            console.print("[yellow]Logs are empty.[/yellow]")
            return ""

        console.print(f"[green]Retrieved {len(logs)} bytes of logs.[/green]")
        return logs
