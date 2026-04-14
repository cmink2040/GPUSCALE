"""Vast.ai provider: deploy bench/vLLM containers and collect results via S3 + SSH."""

from __future__ import annotations

import json
import os
import subprocess
import time

from rich.console import Console

from virt_runner.config import JobConfig
from virt_runner.models import GPUInfo, InferenceEngine
from virt_runner.providers.base import BaseProvider, ProvisionedInstance

console = Console(stderr=True)

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
    console.print(f"[dim]$ {' '.join(cmd[:6])}...[/dim]")
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
    )


class VastProvider(BaseProvider):
    """Deploy benchmarks on Vast.ai.

    Supports two modes:
    - bench image: our container with Meta native / llama.cpp
    - vLLM SSH: deploy vLLM image, SSH in, curl-bash benchmark script
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
        """Search for a matching offer and create an instance."""
        if self.config.engine == InferenceEngine.VLLM and self.config.model_format != "pth":
            return self._provision_vllm()
        return self._provision_bench()

    def _provision_bench(self) -> ProvisionedInstance:
        """Deploy our bench container image."""
        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "RTX_4090"
        offer = self._find_offer(gpu_type, pc.gpu_count, pc.disk_gb, pc.max_dph)

        env_vars = self.config.build_container_env()
        self._run_id = env_vars.get("GPUSCALE_RUN_ID", "")
        env_str = self._build_env_str(env_vars)

        image = self.config.bench_image
        console.print(f"[bold]Creating Vast.ai instance with {image}...[/bold]")

        create_args = [
            "create", "instance", str(offer["id"]),
            "--image", image,
            "--disk", str(pc.disk_gb),
            "--env", env_str,
            "--onstart-cmd", "/app/entrypoint.sh",
            "--raw",
        ]
        instance_id, create_data = self._create_instance(create_args)

        return self._build_instance(instance_id, offer, create_data)

    def _provision_vllm(self) -> ProvisionedInstance:
        """Deploy vLLM image, SSH in later to run benchmark."""
        pc = self.config.provider_config
        gpu_type = pc.gpu_type or "RTX_4090"
        offer = self._find_offer(gpu_type, pc.gpu_count, pc.disk_gb, pc.max_dph)

        env_vars = self.config.build_container_env()
        env_vars["VLLM_API_KEY"] = "gpuscale-bench"
        self._run_id = env_vars.get("GPUSCALE_RUN_ID", "")
        env_str = self._build_env_str(env_vars)

        # Pick vLLM image based on GPU — older GPUs need v0.8.4 (Volta/Turing)
        gpu_name = offer.get("gpu_name", gpu_type).lower()
        older_gpus = ["v100", "tesla", "2080", "3080_ti", "3080 ti", "3070", "2070", "1080"]
        if any(g in gpu_name for g in older_gpus):
            vllm_image = "vllm/vllm-openai:v0.8.4"
            dtype_flag = "--dtype float16"  # Volta doesn't support bfloat16
            console.print(f"[dim]Using vLLM v0.8.4 for older GPU ({gpu_name})[/dim]")
        else:
            vllm_image = "vllm/vllm-openai:latest"
            dtype_flag = "--dtype auto"

        vllm_cmd = (
            f"vllm serve {self.config.model} "
            f"--host 0.0.0.0 --port 8000 "
            f"{dtype_flag} --enforce-eager "
            f"--gpu-memory-utilization 0.95 "
            f"--max-model-len 4096 "
            f"--api-key gpuscale-bench"
        )

        console.print(f"[bold]Creating Vast.ai vLLM instance for {self.config.model}...[/bold]")
        console.print(f"[dim]Image: {vllm_image}[/dim]")

        create_args = [
            "create", "instance", str(offer["id"]),
            "--image", vllm_image,
            "--disk", str(pc.disk_gb),
            "--env", env_str,
            "--onstart-cmd", vllm_cmd,
            "--ssh",
            "--raw",
        ]
        instance_id, create_data = self._create_instance(create_args)

        instance = self._build_instance(instance_id, offer, create_data)
        instance.extra["mode"] = "vllm_ssh"
        return instance

    def wait_ready(self, instance: ProvisionedInstance, timeout_s: int = 1800) -> bool:
        """Wait for instance to be ready, run benchmark if vLLM mode, poll S3 for results."""
        console.print(
            f"[bold]Waiting for instance {instance.instance_id} "
            f"(timeout {timeout_s}s)...[/bold]"
        )
        deadline = time.time() + timeout_s
        poll_interval = 15

        # Phase 1: Wait for instance to be running
        ssh_host = None
        ssh_port = None
        while time.time() < deadline:
            status, ssh_info = self._poll_instance(instance.instance_id)
            if ssh_info:
                ssh_host, ssh_port = ssh_info

            if status == "running":
                console.print(f"  Instance running.")
                break
            elif status in ("exited", "stopped", "offline"):
                console.print(f"[yellow]Instance {status} before benchmark ran.[/yellow]")
                return False
            else:
                console.print(f"  Instance status: [cyan]{status}[/cyan]")
                time.sleep(poll_interval)
        else:
            console.print("[red]Timeout waiting for instance.[/red]")
            return False

        # Phase 2: If vLLM mode, wait for vLLM server to load model, then SSH benchmark
        if instance.extra.get("mode") == "vllm_ssh":
            console.print("[bold]Waiting for vLLM server to start (model download + load)...[/bold]")
            # vLLM needs to: download model from HF, load into GPU, start server
            # This can take 5-15 min depending on model size and network speed
            self._wait_for_vllm_server(instance, deadline)

            if ssh_host and ssh_port:
                self._ssh_run_vllm_bench(instance.instance_id, ssh_host, ssh_port)
            else:
                # Retry polling to get SSH info
                for _ in range(6):
                    status, ssh_info = self._poll_instance(instance.instance_id)
                    if ssh_info:
                        ssh_host, ssh_port = ssh_info
                        self._ssh_run_vllm_bench(instance.instance_id, ssh_host, ssh_port)
                        break
                    time.sleep(10)
                else:
                    console.print("[red]No SSH info available.[/red]")
                    return False

        # Phase 3: Check if SSH captured the results directly (preferred — no S3 dependency)
        ssh_output = getattr(self, "_ssh_output", "")
        if ssh_output and "=== ENGINE_OUTPUT_START ===" in ssh_output:
            console.print(f"[green]Got results from SSH output ({len(ssh_output)} bytes)[/green]")
            instance.extra["logs"] = ssh_output
            return True

        # Phase 4: Poll S3 for the result file. Use whatever budget is left
        # from the job's overall deadline (reserving ~60s for teardown/log
        # fetch), bounded by a 5-min floor and 25-min ceiling. The old
        # hardcoded 300s was fine for small models that fail fast, but a
        # real 27B Q4_K_M run needs ~10 min between "instance running" and
        # "result uploaded to S3" (model pull + 12 iterations + upload),
        # and the hardcoded ceiling silently cut large-model runs short.
        remaining = int(deadline - time.time())
        s3_timeout = max(300, min(remaining - 60, 1500))
        logs = self._fetch_results_from_s3(timeout_s=s3_timeout)
        if logs:
            instance.extra["logs"] = logs
            return True

        # Fallback: try vastai logs
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

    def _find_offer(self, gpu_type: str, gpu_count: int, disk_gb: int, max_dph: float) -> dict:
        """Search Vast.ai for a matching offer."""
        use_community = self.config.provider_config.extra.get("community", False)
        cloud_label = "community" if use_community else "datacenter"
        console.print(f"[bold]Searching Vast.ai for {gpu_count}x {gpu_type} ({cloud_label})...[/bold]")
        query = (
            f"gpu_name={gpu_type} "
            f"num_gpus={gpu_count} "
            f"disk_space>={disk_gb} "
            f"dph<={max_dph} "
            f"inet_down>500 "
            f"reliability>0.95 "
            f"geolocation in [US,CA,GB,DE,FR,NL,SE,NO,FI,IE,CZ,RO,IS]"
        )
        if not use_community:
            query += " datacenter=True"
        # Require 32GB VRAM for V100 to avoid 16GB variants
        if "v100" in gpu_type.lower() or "tesla" in gpu_type.lower():
            query += " gpu_ram>=30"
        # Extra filter via env, e.g. VAST_OFFER_FILTER="cuda_max_good>=13.0"
        # Useful to avoid hosts with old drivers that can't forward-compat
        # consumer GPUs like the 4090 (host 1647 is a persistent offender).
        extra_filter = os.environ.get("VAST_OFFER_FILTER", "").strip()
        if extra_filter:
            query += f" {extra_filter}"
        result = _vast_cli(
            "search", "offers", query, "-o", "dph", "--raw",
            api_key=self.api_key,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vastai search failed: {result.stderr}")

        try:
            offers = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse search output:\n{result.stdout[:500]}")

        if not offers:
            raise RuntimeError(
                f"No Vast.ai offers for {gpu_count}x {gpu_type} "
                f"(>={disk_gb}GB disk, <=${max_dph}$/hr, reliability>0.95)."
            )

        offer = min(offers, key=lambda o: o.get("dph_total", float("inf")))
        console.print(
            f"[green]Selected offer {offer['id']}: "
            f"{offer.get('gpu_name', gpu_type)} x{offer.get('num_gpus', gpu_count)} "
            f"@ ${offer.get('dph_total', '?')}/hr[/green]"
        )
        return offer

    def _create_instance(self, create_args: list[str]) -> tuple[str, dict]:
        """Create a Vast.ai instance and return (instance_id, response_data)."""
        result = _vast_cli(*create_args, api_key=self.api_key)
        if result.returncode != 0:
            raise RuntimeError(f"vastai create failed: {result.stderr}\n{result.stdout}")

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse create output:\n{result.stdout[:500]}")

        instance_id = str(data.get("new_contract", data.get("id", "")))
        if not instance_id:
            raise RuntimeError(f"No instance ID in response: {data}")

        console.print(f"[green]Created instance {instance_id}[/green]")
        return instance_id, data

    def _build_instance(self, instance_id: str, offer: dict, create_data: dict) -> ProvisionedInstance:
        """Build a ProvisionedInstance from offer data."""
        gpu_name = offer.get("gpu_name", "unknown")
        gpu_count = offer.get("num_gpus", 1)
        # Vast.ai's offer `gpu_ram` is in MiB (e.g. 24576 for a 24GB 3090), not GB.
        gpu_ram_mib = int(offer.get("gpu_ram", 0) or 0)
        gpu_ram_gb = gpu_ram_mib / 1024.0 if gpu_ram_mib else 0.0

        gpus = [
            GPUInfo(index=i, name=gpu_name, memory_total_mib=gpu_ram_mib)
            for i in range(gpu_count)
        ]

        return ProvisionedInstance(
            instance_id=instance_id,
            ssh_user="root",
            gpus=gpus,
            extra={
                "offer": offer,
                "gpu_name": gpu_name,
                "gpu_count": gpu_count,
                "gpu_vram_gb": gpu_ram_gb,
            },
        )

    def _build_env_str(self, env_vars: dict[str, str]) -> str:
        """Build the -e KEY=VAL string for vastai --env."""
        parts = []
        for k, v in env_vars.items():
            escaped = v.replace("'", "'\\''")
            parts.append(f"-e {k}='{escaped}'")
        return " ".join(parts)

    def _poll_instance(self, instance_id: str) -> tuple[str, tuple[str, int] | None]:
        """Poll instance status. Returns (status, (ssh_host, ssh_port) or None).

        Catches the vastai CLI Python 3.13 bug gracefully.
        """
        result = _vast_cli(
            "show", "instance", instance_id, "--raw",
            api_key=self.api_key,
        )
        if result.returncode != 0:
            # Catch the Python 3.13 TypeError bug in vastai CLI
            stderr = result.stderr.strip()
            if "TypeError" in stderr or "NoneType" in stderr:
                console.print(f"[dim]  vastai CLI bug (ignored)[/dim]")
                return "unknown", None
            return "unknown", None

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return "unknown", None

        status = data.get("actual_status", "unknown")

        # Extract SSH info
        ssh_host = data.get("ssh_host")
        ssh_port = data.get("ssh_port")
        ssh_info = (ssh_host, int(ssh_port)) if ssh_host and ssh_port else None

        return status, ssh_info

    def _ssh_run_vllm_bench(self, instance_id: str, ssh_host: str, ssh_port: int) -> bool:
        """SSH into a Vast.ai instance and curl-bash the vLLM benchmark script."""
        import paramiko

        ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
        console.print(f"[dim]SSH: root@{ssh_host}:{ssh_port}[/dim]")

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            key = None
            if os.path.exists(ssh_key_path):
                key = paramiko.Ed25519Key.from_private_key_file(ssh_key_path)

            # Retry SSH — Vast.ai SSH daemon takes time, and first attempts may be rejected
            connected = False
            for attempt in range(18):
                try:
                    client.connect(ssh_host, port=ssh_port, username="root", pkey=key, timeout=30)
                    connected = True
                    break
                except Exception as e:
                    msg = str(e)
                    if "Authentication failed" in msg:
                        console.print(f"[dim]  SSH attempt {attempt+1}/18: auth rejected, retrying...[/dim]")
                    else:
                        console.print(f"[dim]  SSH attempt {attempt+1}/18: {msg[:60]}[/dim]")
                    time.sleep(10)

            if not connected:
                console.print("[red]Could not SSH into instance.[/red]")
                return False

            console.print("[green]SSH connected. Running benchmark...[/green]")

            # Build env exports — safe vars only
            env_vars = self.config.build_container_env()
            env_vars["VLLM_API_KEY"] = "gpuscale-bench"
            safe_keys = [
                "MODEL", "ENGINE", "GPUSCALE_RUN_ID", "MODEL_FORMAT",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT",
                "S3_BUCKET", "S3_MODEL_KEY", "AWS_DEFAULT_REGION",
                "MAX_TOKENS", "ITERATIONS", "WARMUP", "HF_TOKEN",
                "VLLM_API_KEY",
            ]
            export_parts = []
            for k in safe_keys:
                if k in env_vars:
                    v = env_vars[k].replace("'", "'\\''")
                    export_parts.append(f"export {k}='{v}'")
            env_cmd = " && ".join(export_parts)

            script_url = "https://raw.githubusercontent.com/cmink2040/GPUSCALE/main/bench-container/bench-vllm.py"
            cmd = f"{env_cmd} && curl -sL {script_url} | python3 ; exit"

            console.print(f"[dim]Running: curl | python3[/dim]")

            # Vast.ai SSH is direct — exec_command works (no PTY proxy issue)
            _, stdout, stderr = client.exec_command(cmd, timeout=3600)

            # Capture + stream output
            captured_lines = []
            for line in stdout:
                stripped = line.rstrip()
                console.print(f"[dim]  {stripped}[/dim]")
                captured_lines.append(stripped)

            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                err = stderr.read().decode()
                console.print(f"[yellow]Script exited with code {exit_code}[/yellow]")
                if err:
                    console.print(f"[dim]{err[-300:]}[/dim]")

            client.close()

            # Store captured output so wait_ready can use it as result
            self._ssh_output = "\n".join(captured_lines)
            return True

        except Exception as exc:
            console.print(f"[red]SSH execution failed: {exc}[/red]")
            return False

    def _wait_for_vllm_server(self, instance: ProvisionedInstance, deadline: float) -> None:
        """SSH into the instance and poll until vLLM server responds on localhost:8000."""
        import paramiko

        ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")

        # First get SSH info
        ssh_host = None
        ssh_port = None
        for _ in range(12):
            _, ssh_info = self._poll_instance(instance.instance_id)
            if ssh_info:
                ssh_host, ssh_port = ssh_info
                break
            time.sleep(10)

        if not ssh_host:
            console.print("[red]Could not get SSH info for vLLM server check.[/red]")
            return

        # Wait for SSH to be ready, then poll vLLM health
        time.sleep(60)  # Initial wait for SSH daemon
        console.print(f"[dim]Checking vLLM at {ssh_host}:{ssh_port}...[/dim]")

        while time.time() < deadline:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                key = None
                if os.path.exists(ssh_key_path):
                    key = paramiko.Ed25519Key.from_private_key_file(ssh_key_path)
                client.connect(ssh_host, port=ssh_port, username="root", pkey=key, timeout=15)

                _, stdout, _ = client.exec_command("curl -s http://localhost:8000/health 2>/dev/null || echo NOTREADY", timeout=10)
                result = stdout.read().decode().strip()
                client.close()

                if result and result != "NOTREADY":
                    console.print(f"[green]vLLM server is ready![/green]")
                    return
                else:
                    # Check if model is downloading via nvidia-smi
                    client2 = paramiko.SSHClient()
                    client2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    client2.connect(ssh_host, port=ssh_port, username="root", pkey=key, timeout=15)
                    _, stdout2, _ = client2.exec_command("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0", timeout=10)
                    vram = stdout2.read().decode().strip()
                    client2.close()
                    console.print(f"[dim]  vLLM not ready yet (GPU VRAM: {vram} MiB)...[/dim]")

            except Exception as e:
                console.print(f"[dim]  SSH check: {str(e)[:60]}[/dim]")

            time.sleep(15)

        console.print("[yellow]vLLM server did not become ready in time.[/yellow]")

    def _fetch_results_from_s3(self, timeout_s: int = 300) -> str:
        """Poll S3 for the result file matching this run's ID."""
        import boto3

        s3 = self.config.s3
        run_id = getattr(self, "_run_id", "")
        if not s3.bucket or not s3.access_key or not run_id:
            return ""

        result_key = f"results/{run_id}.txt"
        console.print(f"[bold]Checking S3 for s3://{s3.bucket}/{result_key}...[/bold]")

        client = boto3.client(
            "s3",
            endpoint_url=s3.endpoint,
            aws_access_key_id=s3.access_key,
            aws_secret_access_key=s3.secret_key,
            region_name=os.getenv("WASABI_REGION", "us-east-1"),
        )

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                result = client.get_object(Bucket=s3.bucket, Key=result_key)
                logs = result["Body"].read().decode()
                if logs:
                    console.print(f"[green]Found result ({len(logs)} bytes)[/green]")
                    return logs
            except Exception:
                pass
            time.sleep(15)

        return ""

    def _fetch_logs(self, instance_id: str) -> str:
        """Fetch container logs via vastai CLI (fallback)."""
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
