"""Pydantic models matching the benchmark_results DB schema."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Provider(str, Enum):
    LOCAL = "local"
    VAST_AI = "vast.ai"
    RUNPOD = "runpod"


class Engine(str, Enum):
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"


class BenchmarkResultCreate(BaseModel):
    """Schema for submitting a new benchmark result. Validated before DB insert."""

    # --- GPU ---
    gpu_name: str = Field(..., min_length=1, max_length=128, examples=["NVIDIA RTX 4090"])
    gpu_vram_gb: float = Field(..., gt=0, le=1000)
    gpu_count: int = Field(..., ge=1, le=64)

    # --- Environment ---
    provider: Provider
    engine: Engine

    # --- Model ---
    model_name: str = Field(..., min_length=1, max_length=256, examples=["meta-llama/Llama-3.1-8B-Instruct"])
    quantization: str = Field(..., min_length=1, max_length=32, examples=["Q4_K_M", "GPTQ-4bit", "FP16"])

    # --- Workload ---
    workload_version: str = Field(..., min_length=1, max_length=32)
    workload_config: dict[str, Any] | None = None

    # --- Performance metrics ---
    tokens_per_sec: float = Field(..., gt=0)
    time_to_first_token_ms: float = Field(..., ge=0)
    prompt_eval_tokens_per_sec: float | None = Field(default=None, ge=0)

    # --- Resource metrics ---
    peak_vram_mb: float | None = Field(default=None, ge=0)
    avg_power_draw_w: float | None = Field(default=None, ge=0)
    peak_power_draw_w: float | None = Field(default=None, ge=0)
    avg_gpu_util_pct: float | None = Field(default=None, ge=0, le=100)
    avg_gpu_temp_c: float | None = Field(default=None, ge=0, le=150)
    total_wall_time_s: float | None = Field(default=None, gt=0)

    # --- Versions ---
    engine_version: str | None = Field(default=None, max_length=64)

    # --- Host (local only) ---
    host_os: str | None = Field(default=None, max_length=128)
    host_kernel: str | None = Field(default=None, max_length=128)
    host_driver_version: str | None = Field(default=None, max_length=64)

    # --- Container ---
    container_image: str | None = Field(default=None, max_length=256)
    container_driver_version: str | None = Field(default=None, max_length=64)

    # --- Raw ---
    raw_output: dict[str, Any] | None = None

    @field_validator("gpu_name")
    @classmethod
    def strip_gpu_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("model_name")
    @classmethod
    def strip_model_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("peak_power_draw_w")
    @classmethod
    def peak_gte_avg_power(cls, v: float | None, info: Any) -> float | None:
        avg = info.data.get("avg_power_draw_w")
        if v is not None and avg is not None and v < avg:
            raise ValueError("peak_power_draw_w must be >= avg_power_draw_w")
        return v

    def to_insert_dict(self) -> dict[str, Any]:
        """Return a dict suitable for Supabase insert (no id/created_at, enums as strings)."""
        data = self.model_dump(exclude_none=True)
        # Convert enums to plain strings for JSON serialization
        data["provider"] = self.provider.value
        data["engine"] = self.engine.value
        return data


class BenchmarkResultRow(BenchmarkResultCreate):
    """A full row as returned from the database, including server-generated fields."""

    id: UUID
    created_at: datetime
    flagged: bool = False

    class Config:
        from_attributes = True
