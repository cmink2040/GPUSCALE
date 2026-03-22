"""Vast.ai provider: provision and manage GPU instances via the vastai CLI."""

from __future__ import annotations

import json
import os
import subprocess
import time

from rich.console import Console

from virt_runner.config import JobConfig
from virt_runner.providers.base import BaseProvider, ProvisionedInstance

console = Console(stderr=True)


def _vast_cli(*args: str, api_key: str = "") -> subprocess.CompletedProcess[str]:
    """Run a vastai CLI command, injecting the API key."""
    env = os.environ.copy()
    if api_key:
        env["VAST_API_KEY"] = api_key
    cmd = ["vastai", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)


class VastProvider(BaseProvider):
    """Provision GPU instances on Vast.ai using the vastai CLI."""

    def __init__(self, config: JobConfig) -> None:
        super().__init__(config)
        self.api_key = config.provider_config.api_key or os.getenv("VAST_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Vast.ai API key required. Set VAST_API_KEY env var or pass via provider_config."
            )

    def provision(self) -> ProvisionedInstance:
        """Search for a matching offer and create an instance."""
        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "RTX_4090"
        gpu_count = pc.gpu_count

        # Search for offers matching the GPU requirements
        console.print(f"[bold]Searching Vast.ai for {gpu_count}x {gpu_type}...[/bold]")
        search_result = _vast_cli(
            "search", "offers",
            f"gpu_name={gpu_type} num_gpus={gpu_count} disk_space>={pc.disk_gb}",
            "--raw",
            api_key=self.api_key,
        )
        if search_result.returncode != 0:
            raise RuntimeError(f"vastai search failed: {search_result.stderr}")

        offers = json.loads(search_result.stdout)
        if not offers:
            raise RuntimeError(
                f"No Vast.ai offers found for {gpu_count}x {gpu_type} "
                f"with >= {pc.disk_gb}GB disk"
            )

        # Pick the cheapest offer by dph_total (dollars per hour)
        offer = min(offers, key=lambda o: o.get("dph_total", float("inf")))
        offer_id = offer["id"]
        console.print(
            f"[green]Selected offer {offer_id}: "
            f"{offer.get('gpu_name', '?')} x{offer.get('num_gpus', '?')} "
            f"@ ${offer.get('dph_total', '?')}/hr[/green]"
        )

        # Create the instance
        image = pc.image or "nvidia/cuda:12.2.0-devel-ubuntu22.04"
        create_result = _vast_cli(
            "create", "instance", str(offer_id),
            "--image", image,
            "--disk", str(pc.disk_gb),
            "--raw",
            api_key=self.api_key,
        )
        if create_result.returncode != 0:
            raise RuntimeError(f"vastai create instance failed: {create_result.stderr}")

        create_data = json.loads(create_result.stdout)
        instance_id = str(create_data.get("new_contract", create_data.get("id", "")))
        console.print(f"[green]Created instance {instance_id}[/green]")

        return ProvisionedInstance(
            instance_id=instance_id,
            ssh_user="root",
            ssh_key_path=self.config.ssh_key_path,
            extra={"offer": offer, "create_response": create_data},
        )

    def wait_ready(self, instance: ProvisionedInstance, timeout_s: int = 300) -> bool:
        """Poll until the Vast.ai instance is running and SSH is available."""
        console.print(f"[bold]Waiting for instance {instance.instance_id} to be ready...[/bold]")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            result = _vast_cli(
                "show", "instance", instance.instance_id,
                "--raw",
                api_key=self.api_key,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                status = data.get("actual_status", "")
                if status == "running":
                    ssh_host = data.get("ssh_host", "")
                    ssh_port = data.get("ssh_port", 22)
                    if ssh_host:
                        instance.host = ssh_host
                        instance.port = int(ssh_port)
                        console.print(
                            f"[green]Instance ready: {ssh_host}:{ssh_port}[/green]"
                        )
                        return True
            time.sleep(10)

        console.print("[red]Timeout waiting for instance to become ready.[/red]")
        return False

    def teardown(self, instance: ProvisionedInstance) -> None:
        """Destroy the Vast.ai instance."""
        if not instance.instance_id or instance.instance_id == "local":
            return
        console.print(f"[bold]Tearing down instance {instance.instance_id}...[/bold]")
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
