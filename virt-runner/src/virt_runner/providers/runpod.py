"""RunPod provider: provision and manage GPU pods via the RunPod HTTP API."""

from __future__ import annotations

import os
import time

import httpx
from rich.console import Console

from virt_runner.config import JobConfig
from virt_runner.providers.base import BaseProvider, ProvisionedInstance

console = Console(stderr=True)

RUNPOD_API_BASE = "https://api.runpod.io/graphql"


class RunPodProvider(BaseProvider):
    """Provision GPU pods on RunPod using their GraphQL API."""

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
        """Create a RunPod GPU pod."""
        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "NVIDIA RTX 4090"
        gpu_count = pc.gpu_count
        image = pc.image or "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04"

        console.print(f"[bold]Creating RunPod pod: {gpu_count}x {gpu_type}...[/bold]")

        mutation = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
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
                "containerDiskInGb": 20,
                "startSsh": True,
                "dockerArgs": "",
            }
        }

        data = self._graphql(mutation, variables)
        pod = data.get("podFindAndDeployOnDemand", {})
        pod_id = pod.get("id", "")
        if not pod_id:
            raise RuntimeError(f"Failed to create RunPod pod. Response: {data}")

        console.print(f"[green]Created pod {pod_id}[/green]")
        return ProvisionedInstance(
            instance_id=pod_id,
            ssh_user="root",
            ssh_key_path=self.config.ssh_key_path,
            extra={"pod": pod},
        )

    def wait_ready(self, instance: ProvisionedInstance, timeout_s: int = 300) -> bool:
        """Poll until the RunPod pod is running and SSH is available."""
        console.print(f"[bold]Waiting for pod {instance.instance_id} to be ready...[/bold]")
        query = """
        query pod($input: PodFilter!) {
            pod(input: $input) {
                id
                desiredStatus
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                }
            }
        }
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            data = self._graphql(query, {"input": {"podId": instance.instance_id}})
            pod = data.get("pod", {})
            runtime = pod.get("runtime")
            if runtime:
                ports = runtime.get("ports", [])
                for port_info in ports:
                    if port_info.get("privatePort") == 22 and port_info.get("isIpPublic"):
                        instance.host = port_info["ip"]
                        instance.port = int(port_info["publicPort"])
                        console.print(
                            f"[green]Pod ready: {instance.host}:{instance.port}[/green]"
                        )
                        return True
            time.sleep(10)

        console.print("[red]Timeout waiting for pod to become ready.[/red]")
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
