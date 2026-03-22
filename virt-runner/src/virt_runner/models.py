"""Pydantic models for benchmark results, metrics, and host metadata."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InferenceEngine(str, Enum):
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"


class Provider(str, Enum):
    LOCAL = "local"
    VAST = "vast"
    RUNPOD = "runpod"


# ---------------------------------------------------------------------------
# Host metadata
# ---------------------------------------------------------------------------


class HostMetadata(BaseModel):
    """System-level metadata collected from the host running the benchmark."""

    hostname: str = ""
    os: str = ""  # e.g. "Linux", "Darwin"
    distro: str = ""  # e.g. "Ubuntu 22.04"
    kernel_version: str = ""
    gpu_driver_version: str = ""
    docker_runtime_version: str = ""
    cuda_version: str = ""


class GPUInfo(BaseModel):
    """Information about a single GPU detected on the host."""

    index: int
    name: str
    uuid: str = ""
    memory_total_mib: int = 0
    driver_version: str = ""
    cuda_version: str = ""
    pci_bus_id: str = ""


# ---------------------------------------------------------------------------
# Workload definition
# ---------------------------------------------------------------------------


class GenerationParams(BaseModel):
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


class PromptMessage(BaseModel):
    role: str = "user"
    content: str = ""


class WorkloadConfig(BaseModel):
    """Standardized benchmark workload configuration."""

    workload_version: str = "1.0"
    prompts: list[PromptMessage] = Field(default_factory=list)
    generation_params: GenerationParams = Field(default_factory=GenerationParams)
    iterations: int = 5
    warmup_iterations: int = 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class IterationMetrics(BaseModel):
    """Metrics for a single benchmark iteration."""

    iteration: int
    tokens_per_sec: float = 0.0
    time_to_first_token_ms: float = 0.0
    prompt_eval_rate_tokens_per_sec: float = 0.0
    peak_vram_mib: float = 0.0
    power_draw_avg_w: float = 0.0
    power_draw_peak_w: float = 0.0
    gpu_utilization_pct: float = 0.0
    gpu_temperature_c: float = 0.0
    wall_time_s: float = 0.0
    tokens_generated: int = 0


class AggregateMetrics(BaseModel):
    """Aggregated metrics across all (non-warmup) iterations."""

    tokens_per_sec_mean: float = 0.0
    tokens_per_sec_std: float = 0.0
    ttft_mean_ms: float = 0.0
    ttft_std_ms: float = 0.0
    prompt_eval_rate_mean: float = 0.0
    peak_vram_mib: float = 0.0
    power_draw_avg_w: float = 0.0
    power_draw_peak_w: float = 0.0
    gpu_utilization_pct_mean: float = 0.0
    gpu_temperature_c_max: float = 0.0
    wall_time_total_s: float = 0.0


# ---------------------------------------------------------------------------
# Benchmark result (top-level)
# ---------------------------------------------------------------------------


class BenchmarkResult(BaseModel):
    """Complete result payload for a benchmark run."""

    run_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider: Provider = Provider.LOCAL
    engine: InferenceEngine = InferenceEngine.LLAMA_CPP
    model: str = ""
    gpu_name: str = ""
    gpu_count: int = 1
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    host: HostMetadata = Field(default_factory=HostMetadata)
    gpus: list[GPUInfo] = Field(default_factory=list)
    iterations: list[IterationMetrics] = Field(default_factory=list)
    aggregate: AggregateMetrics = Field(default_factory=AggregateMetrics)
    raw_engine_output: str = ""
    errors: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
