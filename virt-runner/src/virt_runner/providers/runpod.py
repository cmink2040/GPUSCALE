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
        """Create a RunPod GPU pod. Routes to vLLM pod if engine is vllm + HF model."""
        from virt_runner.models import InferenceEngine

        if self.config.engine == InferenceEngine.VLLM and self.config.model_format != "pth":
            return self._provision_vllm()
        return self._provision_bench()

    def _provision_vllm(self) -> ProvisionedInstance:
        """Deploy vllm/vllm-openai pod, SSH in, curl-bash the benchmark script."""
        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "NVIDIA GeForce RTX 4090"
        gpu_count = pc.gpu_count

        # Build env vars — S3 creds for result upload + model config
        env_vars = self.config.build_container_env()
        self._run_id = env_vars.get("GPUSCALE_RUN_ID", "")

        env_list = [{"key": k, "value": v} for k, v in env_vars.items()]

        console.print(f"[bold]Creating vLLM pod: {gpu_count}x {gpu_type}...[/bold]")
        console.print(f"[dim]Image: vllm/vllm-openai:latest[/dim]")
        console.print(f"[dim]Model: {self.config.model}[/dim]")

        mutation = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                gpuCount
                machine { gpuDisplayName }
            }
        }
        """
        pod_input: dict = {
            "name": f"gpuscale-vllm-{gpu_type.replace(' ', '-').lower()}",
            "imageName": "vllm/vllm-openai:latest",
            "templateId": "iqilnw0ymf",  # RunPod vLLM template (pre-cached image)
            "gpuTypeId": gpu_type,
            "gpuCount": gpu_count,
            "containerDiskInGb": 60,
            "volumeInGb": 50,
            "volumeMountPath": "/workspace",
            "startSsh": True,
            "env": env_list,
            "dockerArgs": self.config.model,  # becomes: vllm serve Qwen/Qwen3.5-9B
        }

        # Try requested GPU, then fallbacks
        gpu_fallbacks = [
            gpu_type, "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
            "NVIDIA RTX A6000", "NVIDIA GeForce RTX 4090",
        ]
        seen = set()
        gpu_fallbacks = [g for g in gpu_fallbacks if not (g in seen or seen.add(g))]

        pod = {}
        for try_gpu in gpu_fallbacks:
            try:
                pod_input["gpuTypeId"] = try_gpu
                pod_input["name"] = f"gpuscale-vllm-{try_gpu.replace(' ', '-').lower()}"
                console.print(f"[dim]Trying {try_gpu}...[/dim]")
                data = self._graphql(mutation, {"input": pod_input})
                pod = data.get("podFindAndDeployOnDemand", {})
                if pod.get("id"):
                    gpu_type = try_gpu
                    break
            except RuntimeError as e:
                if "SUPPLY_CONSTRAINT" in str(e) or "does not have the resources" in str(e):
                    console.print(f"[dim]  No supply for {try_gpu}[/dim]")
                    continue
                raise

        pod_id = pod.get("id", "")
        if not pod_id:
            raise RuntimeError("No GPU available for vLLM pod.")

        gpu_display = pod.get("machine", {}).get("gpuDisplayName", gpu_type) if pod.get("machine") else gpu_type
        console.print(f"[green]Created vLLM pod {pod_id} ({gpu_display})[/green]")

        gpus = [GPUInfo(index=i, name=gpu_display) for i in range(gpu_count)]

        instance = ProvisionedInstance(
            instance_id=pod_id,
            gpus=gpus,
            extra={
                "gpu_name": gpu_display,
                "gpu_count": gpu_count,
                "mode": "vllm_ssh",
            },
        )
        return instance

    def _provision_bench(self) -> ProvisionedInstance:
        """Create a RunPod GPU pod with the bench image, env vars, and optional volume."""
        from virt_runner.config import CUDA13_GPUS, DEFAULT_BENCH_IMAGE_CUDA13

        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "NVIDIA RTX 4090"
        gpu_count = pc.gpu_count

        # Auto-select CUDA 13 image for Blackwell GPUs
        image = self.config.bench_image
        if gpu_type in CUDA13_GPUS and image == "ghcr.io/cmink2040/gpuscale-bench:latest":
            image = DEFAULT_BENCH_IMAGE_CUDA13
            console.print(f"[bold]Blackwell GPU detected — using CUDA 13 image[/bold]")

        # Build env vars for the container
        env_vars = self.config.build_container_env()

        # If a network volume is attached, tell the container to look for models there
        if self.volume_id:
            env_vars["VOLUME_MOUNT_PATH"] = VOLUME_MOUNT_PATH
            console.print(f"[bold]Attaching network volume {self.volume_id}[/bold]")

        # Store run ID for S3 result polling
        self._run_id = env_vars.get("GPUSCALE_RUN_ID", "")

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
            "startSsh": True,
            "env": env_list,
            "dockerArgs": "/app/entrypoint.sh",
            "cloudType": "SECURE",
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

        # Store pod_id so the container can use it in the S3 result key
        # We need to update the env vars on the already-created pod — not possible.
        # Instead, store it for the poller to use.
        self._current_pod_id = pod_id

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
        """Wait for pod to start, run benchmark (SSH if vLLM mode), poll S3 for results."""
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
                console.print(f"  Container running (uptime: {uptime}s)")
                break
            else:
                console.print(f"  Pod status: [cyan]{status}[/cyan] (waiting for container to start)")
                time.sleep(poll_interval)
        else:
            console.print("[red]Timeout waiting for container to start.[/red]")
            return False

        # Phase 2: If vLLM SSH mode, SSH in and run the benchmark script
        if instance.extra.get("mode") == "vllm_ssh":
            console.print("[bold]Waiting for SSH to be ready...[/bold]")
            time.sleep(60)  # vLLM template needs time to start SSH daemon
            if not self._ssh_run_vllm_bench(instance.instance_id):
                console.print("[yellow]SSH benchmark may have failed, checking S3 anyway...[/yellow]")

        # Phase 3: Poll S3 for results
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

    def _ssh_run_vllm_bench(self, pod_id: str) -> bool:
        """SSH into a RunPod pod and curl-bash the vLLM benchmark script."""
        import paramiko

        proxy_host = f"{pod_id}-ssh.proxy.runpod.io"
        ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")

        console.print(f"[dim]SSH: root@{proxy_host}[/dim]")

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            key = None
            if os.path.exists(ssh_key_path):
                key = paramiko.Ed25519Key.from_private_key_file(ssh_key_path)

            # Retry SSH connection — vLLM template needs time to start SSH daemon
            connected = False
            for attempt in range(12):
                try:
                    client.connect(proxy_host, port=22, username="root", pkey=key, timeout=30)
                    connected = True
                    break
                except Exception as e:
                    console.print(f"[dim]  SSH attempt {attempt+1}/12: {e}[/dim]")
                    time.sleep(15)

            if not connected:
                console.print("[red]Could not SSH into pod.[/red]")
                return False

            console.print("[green]SSH connected. Running benchmark...[/green]")

            # Build the env export string for the benchmark script
            env_vars = self.config.build_container_env()
            env_exports = " ".join(f'{k}="{v}"' for k, v in env_vars.items())

            # curl-bash the script with env vars
            script_url = "https://raw.githubusercontent.com/cmink2040/GPUSCALE/main/bench-container/bench-vllm.sh"
            cmd = f'export {env_exports} && curl -sL {script_url} | bash'

            console.print(f"[dim]Running: curl | bash with {len(env_vars)} env vars[/dim]")

            _, stdout, stderr = client.exec_command(cmd, timeout=1800)

            # Stream output
            while True:
                line = stdout.readline()
                if not line:
                    break
                console.print(f"[dim]  {line.rstrip()}[/dim]")

            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                err = stderr.read().decode()
                console.print(f"[red]Script exited with code {exit_code}[/red]")
                console.print(f"[dim]{err[-500:]}[/dim]")

            client.close()
            return exit_code == 0

        except Exception as exc:
            console.print(f"[red]SSH execution failed: {exc}[/red]")
            return False

    def _fetch_logs(self, pod_id: str) -> str:
        """Fetch benchmark results from S3.

        The container uploads results to s3://<bucket>/results/<run_id>.txt
        where run_id is the unique GPUSCALE_RUN_ID passed via env var.
        We poll for that exact key.
        """
        import boto3

        s3 = self.config.s3
        if not s3.bucket or not s3.access_key:
            console.print("[red]No S3 config — cannot retrieve results.[/red]")
            return ""

        run_id = getattr(self, "_run_id", "")
        result_key = f"results/{run_id}.txt" if run_id else ""

        client = boto3.client(
            "s3",
            endpoint_url=s3.endpoint,
            aws_access_key_id=s3.access_key,
            aws_secret_access_key=s3.secret_key,
            region_name=os.getenv("WASABI_REGION", "us-east-1"),
        )

        if result_key:
            console.print(f"[bold]Waiting for s3://{s3.bucket}/{result_key}...[/bold]")
        else:
            console.print(f"[bold]Waiting for results in s3://{s3.bucket}/results/...[/bold]")

        deadline = time.time() + 1800
        while time.time() < deadline:
            # Check pod status
            try:
                data = self._graphql(
                    'query { pod(input: {podId: "%s"}) { id desiredStatus runtime { uptimeInSeconds } } }' % pod_id
                )
                pod = data.get("pod")
                if pod:
                    rt = pod.get("runtime")
                    uptime = rt.get("uptimeInSeconds", 0) if rt else 0
                    console.print(f"[dim]  Pod uptime: {uptime}s, checking S3...[/dim]")
            except Exception:
                pass

            # Try to fetch the exact result key
            if result_key:
                try:
                    result = client.get_object(Bucket=s3.bucket, Key=result_key)
                    logs = result["Body"].read().decode()
                    if logs:
                        console.print(f"[green]Found result: {result_key} ({len(logs)} bytes)[/green]")
                        # Delete after retrieval so it doesn't collide with future runs
                        try:
                            client.delete_object(Bucket=s3.bucket, Key=result_key)
                            console.print(f"[dim]  Cleaned up {result_key} from S3[/dim]")
                        except Exception:
                            pass
                        return logs
                except client.exceptions.NoSuchKey:
                    pass
                except Exception as exc:
                    console.print(f"[dim]  S3 poll error: {exc}[/dim]")

            time.sleep(15)

        console.print("[red]Timeout waiting for results in S3.[/red]")
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
