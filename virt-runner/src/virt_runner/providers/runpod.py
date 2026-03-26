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

        # If using a network volume, try multiple GPU types in case the primary isn't available
        gpu_fallbacks = [
            gpu_type,
            "NVIDIA RTX A5000",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA RTX A6000",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX A4000",
            "NVIDIA GeForce RTX 3080",
        ]
        # Deduplicate while preserving order
        seen = set()
        gpu_fallbacks = [g for g in gpu_fallbacks if not (g in seen or seen.add(g))]

        pod = {}
        last_error = None
        for try_gpu in gpu_fallbacks:
            try:
                pod_input["gpuTypeId"] = try_gpu
                pod_input["name"] = f"gpuscale-bench-{try_gpu.replace(' ', '-').lower()}"
                console.print(f"[dim]Trying {try_gpu}...[/dim]")
                data = self._graphql(mutation, {"input": pod_input})
                pod = data.get("podFindAndDeployOnDemand", {})
                if pod.get("id"):
                    gpu_type = try_gpu  # Update for display
                    break
            except RuntimeError as e:
                last_error = e
                if "SUPPLY_CONSTRAINT" in str(e) or "does not have the resources" in str(e):
                    console.print(f"[dim]  No supply for {try_gpu}[/dim]")
                    continue
                raise

        pod_id = pod.get("id", "")
        if not pod_id:
            raise RuntimeError(
                f"No GPU available in the volume's datacenter. Last error: {last_error}"
            )

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
        """Wait for the container to start, then SSH in to retrieve results.

        The container runs the benchmark and writes results to /workspace/gpuscale_result.txt,
        then stays alive so we can SSH in. We poll until the container is running,
        then use _fetch_logs (which SSHes in) to get the results.
        """
        console.print(
            f"[bold]Waiting for pod {instance.instance_id} "
            f"to start (timeout {timeout_s}s)...[/bold]"
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

        # Phase 1: Wait for container to actually start (runtime != null)
        while time.time() < deadline:
            try:
                data = self._graphql(query, {"input": {"podId": instance.instance_id}})
            except Exception as exc:
                console.print(f"[yellow]Poll error: {exc}[/yellow]")
                time.sleep(poll_interval)
                continue

            pod = data.get("pod")
            if pod is None:
                console.print("[yellow]Pod not found.[/yellow]")
                return False

            status = pod.get("desiredStatus", "unknown")
            runtime = pod.get("runtime")

            if runtime is not None:
                uptime = runtime.get("uptimeInSeconds", 0)
                console.print(f"  Container running (uptime: {uptime}s). Fetching results via SSH...")
                break
            else:
                console.print(f"  Pod status: [cyan]{status}[/cyan] (waiting for container to start)")
                time.sleep(poll_interval)
        else:
            console.print("[red]Timeout waiting for container to start.[/red]")
            return False

        # Phase 2: SSH in and wait for benchmark to finish, then get results
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
        """Fetch benchmark results from a RunPod pod via SSH.

        RunPod has no logs API for pods. Instead, the container writes results
        to /workspace/gpuscale_result.txt and stays alive. We SSH in, cat the
        file, and return the contents.
        """
        console.print(f"[bold]Fetching results from pod {pod_id} via SSH...[/bold]")

        # Get SSH connection info
        ssh_info = self._get_ssh_info(pod_id)
        if not ssh_info:
            console.print("[red]Could not get SSH connection info.[/red]")
            return ""

        host, port = ssh_info
        result_file = "/workspace/gpuscale_result.txt"
        done_marker = "/workspace/gpuscale_done"

        import paramiko

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, port=port, username="root", timeout=30)

            # Wait for the benchmark to finish (check for done marker)
            deadline = time.time() + 1800
            while time.time() < deadline:
                _, stdout, _ = client.exec_command(f"test -f {done_marker} && echo DONE || echo WAITING")
                status = stdout.read().decode().strip()
                if status == "DONE":
                    break
                console.print("[dim]  Benchmark still running...[/dim]")
                time.sleep(15)

            # Cat the result file
            _, stdout, stderr = client.exec_command(f"cat {result_file}")
            logs = stdout.read().decode()
            err = stderr.read().decode()

            if logs:
                console.print(f"[green]Retrieved {len(logs)} bytes of results via SSH.[/green]")
            else:
                console.print(f"[yellow]Result file empty or missing. stderr: {err[:200]}[/yellow]")

            client.close()
            return logs

        except Exception as exc:
            console.print(f"[red]SSH fetch failed: {exc}[/red]")
            return ""

    def _get_ssh_info(self, pod_id: str) -> tuple[str, int] | None:
        """Get SSH host and port for a RunPod pod."""
        # Poll until SSH port is available
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                data = self._graphql(
                    'query { pod(input: {podId: "%s"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }' % pod_id
                )
                pod = data.get("pod")
                if not pod or not pod.get("runtime"):
                    time.sleep(5)
                    continue
                ports = pod["runtime"].get("ports", [])
                for p in ports:
                    if p.get("privatePort") == 22 and p.get("isIpPublic"):
                        return (p["ip"], int(p["publicPort"]))
                # Sometimes SSH is on a different port
                for p in ports:
                    if p.get("isIpPublic"):
                        return (p["ip"], int(p["publicPort"]))
            except Exception:
                pass
            time.sleep(5)
        return None

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
