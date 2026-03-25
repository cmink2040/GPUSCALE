"""RunPod provider: deploy bench container as a pod and collect logs via GraphQL API."""

from __future__ import annotations

import json
import os
import time

import httpx
from rich.console import Console

from virt_runner.config import JobConfig
from virt_runner.models import GPUInfo
from virt_runner.providers.base import BaseProvider, ProvisionedInstance

console = Console(stderr=True)

RUNPOD_API_BASE = "https://api.runpod.io/graphql"


class RunPodProvider(BaseProvider):
    """Deploy the benchmark container on RunPod and collect results.

    Flow:
    1. Create a pod with the bench image + env vars
    2. Poll until the pod exits (container runs to completion)
    3. Fetch logs (stdout/stderr with engine output + GPU metrics)
    4. Terminate the pod
    """

    def __init__(self, config: JobConfig) -> None:
        super().__init__(config)
        self.api_key = config.provider_config.api_key or os.getenv("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "RunPod API key required. Set RUNPOD_API_KEY env var or pass via provider_config."
            )
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )

    def _graphql(self, query: str, variables: dict | None = None) -> dict:
        """Execute a GraphQL query against the RunPod API."""
        payload: dict = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self._client.post(RUNPOD_API_BASE, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod API error: {data['errors']}")
        return data.get("data", {})

    def provision(self) -> ProvisionedInstance:
        """Create a RunPod GPU pod with the bench image and env vars."""
        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "NVIDIA RTX 4090"
        gpu_count = pc.gpu_count
        image = self.config.bench_image

        # Build env vars dict for the container
        env_vars = self.config.build_container_env()
        # RunPod expects env as a dict in the dockerArgs or as environment variables
        env_dict = {k: v for k, v in env_vars.items()}

        console.print(f"[bold]Creating RunPod pod: {gpu_count}x {gpu_type}...[/bold]")
        console.print(f"[dim]Image: {image}[/dim]")

        mutation = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                gpuCount
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                }
                machine {
                    gpuDisplayName
                }
            }
        }
        """
        variables = {
            "input": {
                "name": f"gpuscale-bench-{gpu_type.replace(' ', '-').lower()}",
                "imageName": image,
                "gpuTypeId": gpu_type,
                "gpuCount": gpu_count,
                "volumeInGb": pc.disk_gb,
                "containerDiskInGb": 40,
                "startSsh": False,
                "env": env_dict,
            }
        }

        data = self._graphql(mutation, variables)
        pod = data.get("podFindAndDeployOnDemand", {})
        pod_id = pod.get("id", "")
        if not pod_id:
            raise RuntimeError(f"Failed to create RunPod pod. Response: {data}")

        gpu_display = (
            pod.get("machine", {}).get("gpuDisplayName", gpu_type) if pod.get("machine") else gpu_type
        )

        console.print(f"[green]Created pod {pod_id} ({gpu_display})[/green]")

        gpus = [
            GPUInfo(index=i, name=gpu_display)
            for i in range(gpu_count)
        ]

        return ProvisionedInstance(
            instance_id=pod_id,
            gpus=gpus,
            extra={
                "pod": pod,
                "gpu_name": gpu_display,
                "gpu_count": gpu_count,
            },
        )

    def wait_ready(self, instance: ProvisionedInstance, timeout_s: int = 1800) -> bool:
        """Poll until the RunPod pod finishes running the benchmark container.

        Returns True if we got logs, False on timeout.
        """
        console.print(
            f"[bold]Waiting for pod {instance.instance_id} "
            f"to complete (timeout {timeout_s}s)...[/bold]"
        )

        query = """
        query pod($input: PodFilter!) {
            pod(input: $input) {
                id
                desiredStatus
                runtime {
                    uptimeInSeconds
                }
            }
        }
        """
        deadline = time.time() + timeout_s
        poll_interval = 15
        last_status = ""

        while time.time() < deadline:
            try:
                data = self._graphql(query, {"input": {"podId": instance.instance_id}})
            except Exception as exc:
                console.print(f"[yellow]Poll error: {exc}[/yellow]")
                time.sleep(poll_interval)
                continue

            pod = data.get("pod")
            if pod is None:
                # Pod may have already terminated
                console.print("[yellow]Pod not found -- may have already exited.[/yellow]")
                logs = self._fetch_logs(instance.instance_id)
                if logs:
                    instance.extra["logs"] = logs
                    return True
                return False

            status = pod.get("desiredStatus", "unknown")
            runtime = pod.get("runtime")

            if status != last_status:
                console.print(f"  Pod status: [cyan]{status}[/cyan]")
                last_status = status

            # If runtime is None and status is EXITED, the container finished
            if status == "EXITED" or (status == "RUNNING" and runtime is None):
                console.print("[green]Container finished.[/green]")
                logs = self._fetch_logs(instance.instance_id)
                if logs:
                    instance.extra["logs"] = logs
                    return True
                return False

            time.sleep(poll_interval)

        console.print("[red]Timeout waiting for pod to complete.[/red]")
        logs = self._fetch_logs(instance.instance_id)
        if logs:
            instance.extra["logs"] = logs
            return True
        return False

    def teardown(self, instance: ProvisionedInstance) -> None:
        """Terminate and delete the RunPod pod."""
        if not instance.instance_id or instance.instance_id == "local":
            return
        console.print(f"[bold]Terminating pod {instance.instance_id}...[/bold]")
        mutation = """
        mutation terminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        try:
            self._graphql(mutation, {"input": {"podId": instance.instance_id}})
            console.print(f"[green]Pod {instance.instance_id} terminated.[/green]")
        except Exception as exc:
            console.print(f"[red]Warning: teardown failed: {exc}[/red]")

    def get_name(self) -> str:
        return "runpod"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_logs(self, pod_id: str) -> str:
        """Fetch container logs from a RunPod pod."""
        console.print(f"[bold]Fetching logs for pod {pod_id}...[/bold]")
        query = """
        query podLogs($input: PodLogsInput!) {
            podLogs(input: $input)
        }
        """
        try:
            data = self._graphql(query, {"input": {"podId": pod_id}})
            logs = data.get("podLogs", "")
            if logs:
                console.print(f"[green]Retrieved {len(logs)} bytes of logs.[/green]")
            else:
                console.print("[yellow]Logs are empty.[/yellow]")
            return logs or ""
        except Exception as exc:
            console.print(f"[red]Failed to fetch logs: {exc}[/red]")
            return ""
