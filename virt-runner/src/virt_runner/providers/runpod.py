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

# Default mount path for the network volume inside the container
VOLUME_MOUNT_PATH = "/runpod-volume"


class RunPodProvider(BaseProvider):
    """Deploy the benchmark container on RunPod and collect results.

    Supports attaching a Network Volume for persistent model storage.
    Set RUNPOD_VOLUME_ID env var to attach an existing volume.

    Flow:
    1. Create a pod with the bench image + env vars + optional volume
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
        self.volume_id = os.getenv("RUNPOD_VOLUME_ID", "")

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
        """Create a RunPod GPU pod with the bench image, env vars, and optional volume."""
        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "NVIDIA RTX 4090"
        gpu_count = pc.gpu_count
        image = self.config.bench_image

        # Build env vars for the container
        env_vars = self.config.build_container_env()

        # If a network volume is attached, tell the container to look for models there
        if self.volume_id:
            env_vars["VOLUME_MOUNT_PATH"] = VOLUME_MOUNT_PATH
            console.print(f"[bold]Attaching network volume {self.volume_id}[/bold]")

        # RunPod expects env as [{key: "K", value: "V"}, ...]
        env_list = [{"key": k, "value": v} for k, v in env_vars.items()]

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
        pod_input: dict = {
            "name": f"gpuscale-bench-{gpu_type.replace(' ', '-').lower()}",
            "imageName": image,
            "gpuTypeId": gpu_type,
            "gpuCount": gpu_count,
            "containerDiskInGb": 40,
            "startSsh": False,
            "env": env_list,
            "dockerArgs": "/app/entrypoint.sh",
        }

        # Attach network volume if configured
        if self.volume_id:
            pod_input["networkVolumeId"] = self.volume_id
            pod_input["volumeMountPath"] = VOLUME_MOUNT_PATH
        else:
            pod_input["volumeInGb"] = pc.disk_gb
            pod_input["volumeMountPath"] = "/workspace"

        data = self._graphql(mutation, {"input": pod_input})
        pod = data.get("podFindAndDeployOnDemand", {})
        pod_id = pod.get("id", "")
        if not pod_id:
            raise RuntimeError(f"Failed to create RunPod pod. Response: {data}")

        gpu_display = (
            pod.get("machine", {}).get("gpuDisplayName", gpu_type)
            if pod.get("machine")
            else gpu_type
        )

        console.print(f"[green]Created pod {pod_id} ({gpu_display})[/green]")

        gpus = [GPUInfo(index=i, name=gpu_display) for i in range(gpu_count)]

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
        """Poll until the RunPod pod finishes running the benchmark container."""
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
        saw_runtime = False  # Track if the container actually started

        while time.time() < deadline:
            try:
                data = self._graphql(query, {"input": {"podId": instance.instance_id}})
            except Exception as exc:
                console.print(f"[yellow]Poll error: {exc}[/yellow]")
                time.sleep(poll_interval)
                continue

            pod = data.get("pod")
            if pod is None:
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

            # runtime=None + RUNNING means still pulling image / initializing
            # runtime present + RUNNING means container is actively running
            if runtime is not None:
                if not saw_runtime:
                    uptime = runtime.get("uptimeInSeconds", 0)
                    console.print(f"  Container started (uptime: {uptime}s)")
                    saw_runtime = True

            if status == "RUNNING" and runtime is None:
                if saw_runtime:
                    # Runtime disappeared — container exited
                    console.print("[green]Container finished (runtime cleared).[/green]")
                else:
                    # Still initializing / pulling image
                    pass
                time.sleep(poll_interval)
                continue

            if status == "EXITED":
                console.print("[green]Container exited.[/green]")
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
    # Network Volume management
    # ------------------------------------------------------------------

    def create_network_volume(
        self, name: str = "gpuscale-models", size_gb: int = 50, region: str = "US-TX-3"
    ) -> str:
        """Create a RunPod network volume for persistent model storage.

        Returns the volume ID.
        """
        mutation = """
        mutation createVolume($input: CreateNetworkVolumeInput!) {
            createNetworkVolume(input: $input) {
                id
                name
                size
                dataCenterId
            }
        }
        """
        data = self._graphql(mutation, {
            "input": {
                "name": name,
                "size": size_gb,
                "dataCenterId": region,
            }
        })
        volume = data.get("createNetworkVolume", {})
        vol_id = volume.get("id", "")
        if not vol_id:
            raise RuntimeError(f"Failed to create volume. Response: {data}")
        console.print(
            f"[green]Created network volume {vol_id} "
            f"({name}, {size_gb}GB, {region})[/green]"
        )
        return vol_id

    def sync_s3_to_volume(self, volume_id: str) -> None:
        """Sync models from Wasabi S3 to a RunPod network volume.

        Launches a temporary pod that mounts the volume and runs aws s3 sync
        from the Wasabi bucket to the volume.
        """
        s3 = self.config.s3

        if not s3.bucket:
            raise ValueError("S3 bucket not configured. Set WASABI_BUCKET env var.")

        console.print(
            f"[bold]Syncing s3://{s3.bucket} -> RunPod volume {volume_id}...[/bold]"
        )

        # Build a sync script that runs inside a minimal container
        sync_script = (
            f"pip install awscli -q && "
            f"aws configure set aws_access_key_id '{s3.access_key}' && "
            f"aws configure set aws_secret_access_key '{s3.secret_key}' && "
            f"aws configure set default.region '{os.getenv('WASABI_REGION', 'us-east-1')}' && "
            f"aws s3 sync s3://{s3.bucket}/ {VOLUME_MOUNT_PATH}/models/ "
            f"--endpoint-url '{s3.endpoint}' && "
            f"echo 'SYNC_COMPLETE' && "
            f"ls -lhR {VOLUME_MOUNT_PATH}/models/"
        )

        mutation = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
            }
        }
        """
        # RunPod requires a GPU — use cheap/available options
        gpu_options = [
            "NVIDIA RTX A2000",
            "NVIDIA GeForce RTX 3070",
            "NVIDIA RTX A4000",
            "NVIDIA RTX A5000",
            "NVIDIA GeForce RTX 3080",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA RTX 4000 SFF Ada Generation",
            "NVIDIA GeForce RTX 4070 Ti",
            "NVIDIA GeForce RTX 4090",
        ]
        data = None
        for gpu_type in gpu_options:
            try:
                console.print(f"[dim]Trying sync pod with {gpu_type}...[/dim]")
                data = self._graphql(mutation, {
                    "input": {
                        "name": "gpuscale-s3-sync",
                        "imageName": "python:3.11-slim",
                        "gpuTypeId": gpu_type,
                        "gpuCount": 1,
                        "networkVolumeId": volume_id,
                        "volumeMountPath": VOLUME_MOUNT_PATH,
                        "containerDiskInGb": 10,
                        "startSsh": False,
                        "dockerArgs": f"bash -c \"{sync_script}\"",
                    }
                })
                break
            except RuntimeError as e:
                if "SUPPLY_CONSTRAINT" in str(e):
                    continue
                raise
        if data is None:
            raise RuntimeError("No GPU available in this region for sync pod. Try a different region.")

        pod = data.get("podFindAndDeployOnDemand", {})
        pod_id = pod.get("id", "")
        if not pod_id:
            raise RuntimeError(f"Failed to create sync pod. Response: {data}")

        console.print(f"[green]Sync pod created: {pod_id}[/green]")
        console.print("[dim]Waiting for sync to complete...[/dim]")

        # Poll until done
        query = """
        query pod($input: PodFilter!) {
            pod(input: $input) {
                id
                desiredStatus
                runtime { uptimeInSeconds }
            }
        }
        """
        deadline = time.time() + 1800  # 30 min timeout
        while time.time() < deadline:
            try:
                data = self._graphql(query, {"input": {"podId": pod_id}})
                pod_data = data.get("pod")
                if pod_data is None or pod_data.get("desiredStatus") == "EXITED":
                    break
            except Exception:
                pass
            time.sleep(15)

        # Fetch logs
        logs = self._fetch_logs(pod_id)
        if "SYNC_COMPLETE" in logs:
            console.print("[green]S3 sync complete![/green]")
        else:
            console.print(f"[yellow]Sync may not have completed. Logs:[/yellow]\n{logs[-500:]}")

        # Clean up sync pod
        self._terminate_pod(pod_id)

    def list_volumes(self) -> list[dict]:
        """List all RunPod network volumes."""
        query = """
        query {
            myself {
                networkVolumes {
                    id
                    name
                    size
                    dataCenterId
                }
            }
        }
        """
        data = self._graphql(query)
        return data.get("myself", {}).get("networkVolumes", [])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_logs(self, pod_id: str) -> str:
        """Fetch container logs from a RunPod pod via REST API."""
        console.print(f"[bold]Fetching logs for pod {pod_id}...[/bold]")

        # RunPod exposes logs via their REST API
        try:
            resp = self._client.get(
                f"https://api.runpod.io/v2/{pod_id}/status",
            )
            if resp.status_code == 200:
                data = resp.json()
                output = data.get("output", "")
                if output:
                    console.print(f"[green]Retrieved output from REST API.[/green]")
                    return str(output)
        except Exception:
            pass

        # Fallback: try GraphQL with different query formats
        queries = [
            ("query { pod(input: {podId: \"%s\"}) { runtime { logs } } }" % pod_id, None),
        ]

        for query, variables in queries:
            try:
                data = self._graphql(query, variables)
                # Navigate to logs in response
                pod = data.get("pod", {})
                if pod:
                    runtime = pod.get("runtime") or {}
                    logs = runtime.get("logs", "")
                    if logs:
                        console.print(f"[green]Retrieved {len(logs)} bytes of logs.[/green]")
                        return logs
            except Exception as exc:
                console.print(f"[dim]Log query failed: {exc}[/dim]")
                continue

        console.print("[yellow]Could not retrieve logs.[/yellow]")
        return ""

    def _terminate_pod(self, pod_id: str) -> None:
        """Terminate a pod silently."""
        mutation = """
        mutation terminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        try:
            self._graphql(mutation, {"input": {"podId": pod_id}})
        except Exception:
            pass
