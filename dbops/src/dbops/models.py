"""SQLAlchemy ORM model and Pydantic validation models for benchmark_results."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    Numeric,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    # Identity
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # GPU
    gpu_name: Mapped[str] = mapped_column(Text, nullable=False)
    gpu_vram_gb: Mapped[float] = mapped_column(Numeric, nullable=False)
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Environment
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    engine: Mapped[str] = mapped_column(Text, nullable=False)

    # Model
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    quantization: Mapped[str] = mapped_column(Text, nullable=False)

    # Workload
    workload_version: Mapped[str] = mapped_column(Text, nullable=False)
    workload_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Performance metrics
    tokens_per_sec: Mapped[float] = mapped_column(Numeric, nullable=False)
    time_to_first_token_ms: Mapped[float] = mapped_column(Numeric, nullable=False)
    prompt_eval_tokens_per_sec: Mapped[float | None] = mapped_column(Numeric, nullable=True)

    # Resource metrics
    peak_vram_mb: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    avg_power_draw_w: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    peak_power_draw_w: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    avg_gpu_util_pct: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    avg_gpu_temp_c: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    total_wall_time_s: Mapped[float | None] = mapped_column(Numeric, nullable=True)

    # Versions
    engine_version: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Host (local only)
    host_os: Mapped[str | None] = mapped_column(Text, nullable=True)
    host_kernel: Mapped[str | None] = mapped_column(Text, nullable=True)
    host_driver_version: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Container
    container_image: Mapped[str | None] = mapped_column(Text, nullable=True)
    container_driver_version: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Raw output
    raw_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Moderation
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint("gpu_vram_gb > 0", name="ck_gpu_vram_positive"),
        CheckConstraint("gpu_count >= 1 AND gpu_count <= 64", name="ck_gpu_count_range"),
        CheckConstraint("provider IN ('local', 'vast.ai', 'vast.ai (community)', 'runpod')", name="ck_provider_enum"),
        CheckConstraint("engine IN ('llama.cpp', 'vllm')", name="ck_engine_enum"),
        CheckConstraint("tokens_per_sec > 0", name="ck_tps_positive"),
        CheckConstraint("time_to_first_token_ms >= 0", name="ck_ttft_non_negative"),
        CheckConstraint(
            "prompt_eval_tokens_per_sec IS NULL OR prompt_eval_tokens_per_sec >= 0",
            name="ck_prompt_eval_non_negative",
        ),
        CheckConstraint(
            "peak_vram_mb IS NULL OR peak_vram_mb >= 0", name="ck_vram_non_negative"
        ),
        CheckConstraint(
            "avg_power_draw_w IS NULL OR avg_power_draw_w >= 0", name="ck_avg_power_non_negative"
        ),
        CheckConstraint(
            "peak_power_draw_w IS NULL OR peak_power_draw_w >= 0", name="ck_peak_power_non_negative"
        ),
        CheckConstraint(
            "avg_gpu_util_pct IS NULL OR (avg_gpu_util_pct >= 0 AND avg_gpu_util_pct <= 100)",
            name="ck_gpu_util_range",
        ),
        CheckConstraint(
            "avg_gpu_temp_c IS NULL OR (avg_gpu_temp_c >= 0 AND avg_gpu_temp_c <= 150)",
            name="ck_gpu_temp_range",
        ),
        CheckConstraint(
            "total_wall_time_s IS NULL OR total_wall_time_s > 0", name="ck_wall_time_positive"
        ),
        CheckConstraint(
            "peak_power_draw_w IS NULL OR avg_power_draw_w IS NULL "
            "OR peak_power_draw_w >= avg_power_draw_w",
            name="ck_peak_gte_avg_power",
        ),
        Index("ix_br_gpu_name", "gpu_name"),
        Index("ix_br_model_name", "model_name"),
        Index("ix_br_engine", "engine"),
        Index("ix_br_provider", "provider"),
        Index("ix_br_quantization", "quantization"),
        Index("ix_br_created_at", "created_at"),
        Index("ix_br_flagged", "flagged", postgresql_where="flagged = true"),
    )

    def __repr__(self) -> str:
        return (
            f"<BenchmarkResult {self.id} gpu={self.gpu_name} "
            f"model={self.model_name} engine={self.engine} "
            f"tps={self.tokens_per_sec}>"
        )


# ---------------------------------------------------------------------------
# Pydantic validation models (used by CLI before DB insert)
# ---------------------------------------------------------------------------


class Provider(str, Enum):
    LOCAL = "local"
    VAST_AI = "vast.ai"
    VAST_AI_COMMUNITY = "vast.ai (community)"
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
    model_name: str = Field(
        ..., min_length=1, max_length=256, examples=["meta-llama/Llama-3.1-8B-Instruct"]
    )
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

    def to_orm(self) -> BenchmarkResult:
        """Convert validated pydantic model to a SQLAlchemy ORM instance."""
        data = self.model_dump(exclude_none=True)
        data["provider"] = self.provider.value
        data["engine"] = self.engine.value
        return BenchmarkResult(**data)


class BenchmarkResultRow(BenchmarkResultCreate):
    """A full row as returned from the database, including server-generated fields."""

    id: uuid.UUID
    created_at: datetime
    flagged: bool = False

    class Config:
        from_attributes = True
