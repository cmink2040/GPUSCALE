"""Collect local host metadata: OS, kernel, GPU driver, Docker runtime."""

from __future__ import annotations

import platform
import subprocess

from virt_runner.models import GPUInfo, HostMetadata


def _run(cmd: list[str], timeout: int = 10) -> str:
    """Run a command and return stripped stdout, or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _get_distro() -> str:
    """Attempt to read Linux distro info from /etc/os-release."""
    try:
        with open("/etc/os-release") as f:
            info: dict[str, str] = {}
            for line in f:
                if "=" in line:
                    key, _, value = line.strip().partition("=")
                    info[key] = value.strip('"')
            return info.get("PRETTY_NAME", info.get("NAME", ""))
    except FileNotFoundError:
        # macOS or other non-Linux
        if platform.system() == "Darwin":
            ver = _run(["sw_vers", "-productVersion"])
            return f"macOS {ver}" if ver else "macOS"
        return ""


def _get_docker_version() -> str:
    raw = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    return raw or _run(["docker", "--version"])


def _get_nvidia_driver_version() -> str:
    raw = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])
    # nvidia-smi may return one line per GPU; take the first
    return raw.split("\n")[0].strip() if raw else ""


def _get_cuda_version() -> str:
    raw = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    # Attempt to get CUDA version from nvidia-smi header instead
    header = _run(["nvidia-smi"])
    for line in header.split("\n"):
        if "CUDA Version" in line:
            # Line looks like: "| NVIDIA-SMI 535.129.03  Driver Version: 535.129.03  CUDA Version: 12.2  |"
            parts = line.split("CUDA Version:")
            if len(parts) > 1:
                return parts[1].strip().rstrip("|").strip()
    return ""


def collect_host_metadata() -> HostMetadata:
    """Gather host-level metadata for the current machine."""
    return HostMetadata(
        hostname=platform.node(),
        os=platform.system(),
        distro=_get_distro(),
        kernel_version=platform.release(),
        gpu_driver_version=_get_nvidia_driver_version(),
        docker_runtime_version=_get_docker_version(),
        cuda_version=_get_cuda_version(),
    )


def detect_local_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs via nvidia-smi. Returns an empty list if unavailable."""
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        # Try ROCm (AMD)
        return _detect_rocm_gpus()

    gpus: list[GPUInfo] = []
    for line in raw.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        cuda_ver = _get_cuda_version()
        gpus.append(
            GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                uuid=parts[2],
                memory_total_mib=int(float(parts[3])),
                driver_version=parts[4],
                cuda_version=cuda_ver,
                pci_bus_id=parts[5],
            )
        )
    return gpus


def _detect_rocm_gpus() -> list[GPUInfo]:
    """Fallback: detect AMD GPUs via rocm-smi."""
    raw = _run(["rocm-smi", "--showid", "--showproductname", "--csv"])
    if not raw:
        return []
    # Very basic parsing; rocm-smi output varies by version
    gpus: list[GPUInfo] = []
    lines = raw.strip().split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            continue  # skip header
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append(
                GPUInfo(
                    index=i - 1,
                    name=parts[1] if len(parts) > 1 else f"AMD GPU {i - 1}",
                    uuid=parts[0] if parts[0] else "",
                )
            )
    return gpus
