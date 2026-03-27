"""Job configuration model for benchmark runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from virt_runner.models import (
    GenerationParams,
    InferenceEngine,
    PromptMessage,
    Provider,
    WorkloadConfig,
)

DEFAULT_WORKLOAD_PATH = Path(__file__).resolve().parent.parent.parent / "workloads" / "default.json"
DEFAULT_BENCH_IMAGE = "ghcr.io/cmink2040/gpuscale-bench:latest"
DEFAULT_BENCH_IMAGE_CUDA13 = "ghcr.io/cmink2040/gpuscale-bench:cuda13"
DEFAULT_S3_ENDPOINT = "https://s3.wasabisys.com"

# GPUs that require CUDA 13+ (Blackwell architecture)
CUDA13_GPUS = {"NVIDIA GeForce RTX 5090", "NVIDIA RTX B200", "NVIDIA B200"}


class ProviderConfig(BaseModel):
    """Provider-specific provisioning configuration."""

    api_key: str = ""
    gpu_type: str = ""  # e.g. "RTX_4090", "A100_80GB"
    gpu_count: int = 1
    disk_gb: int = 100
    image: str = ""  # cloud VM image / template id
    region: str = ""
    max_dph: float = 2.0  # max dollars per hour for cloud offers
    extra: dict[str, Any] = Field(default_factory=dict)


class S3Config(BaseModel):
    """S3/Wasabi configuration for model storage."""

    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""
    bucket: str = ""
    model_key: str = ""  # object key within bucket, e.g. "models/llama-7b.gguf"

    @model_validator(mode="after")
    def _fill_from_env(self) -> "S3Config":
        if not self.endpoint:
            self.endpoint = os.getenv("WASABI_ENDPOINT", os.getenv("S3_ENDPOINT", DEFAULT_S3_ENDPOINT))
        if not self.access_key:
            self.access_key = os.getenv("AWS_ACCESS_KEY_ID", os.getenv("WASABI_ACCESS_KEY", ""))
        if not self.secret_key:
            self.secret_key = os.getenv(
                "AWS_SECRET_ACCESS_KEY", os.getenv("WASABI_SECRET_KEY", "")
            )
        if not self.bucket:
            self.bucket = os.getenv("S3_BUCKET", os.getenv("WASABI_BUCKET", ""))
        if not self.model_key:
            self.model_key = os.getenv("S3_MODEL_KEY", "")
        return self


class JobConfig(BaseModel):
    """Top-level benchmark job configuration."""

    provider: Provider = Provider.LOCAL
    engine: InferenceEngine = InferenceEngine.LLAMA_CPP
    model: str = ""  # model identifier, e.g. "meta-llama/Llama-3.1-8B-Instruct"
    model_format: str = ""  # "full", "gguf", "gptq" -- auto-detected if empty
    gguf_quant: str = ""  # e.g. "Q4_K_M"
    bench_image: str = DEFAULT_BENCH_IMAGE
    provider_config: ProviderConfig = Field(default_factory=ProviderConfig)
    s3: S3Config = Field(default_factory=S3Config)
    workload: WorkloadConfig | None = None
    ssh_key_path: str = ""
    ssh_user: str = "root"
    timeout_s: int = 1800  # 30 min default
    hf_token: str = ""

    @model_validator(mode="after")
    def _fill_from_env(self) -> "JobConfig":
        """Populate API keys and tokens from environment variables if not set."""
        # Provider API keys
        if not self.provider_config.api_key:
            env_map = {
                Provider.VAST: "VAST_API_KEY",
                Provider.RUNPOD: "RUNPOD_API_KEY",
            }
            env_var = env_map.get(self.provider, "")
            if env_var:
                self.provider_config.api_key = os.getenv(env_var, "")

        # HuggingFace token
        if not self.hf_token:
            self.hf_token = os.getenv("HF_TOKEN", "")

        return self

    def load_workload(self, path: Path | None = None) -> WorkloadConfig:
        """Load workload from a JSON file, falling back to the bundled default."""
        if self.workload is not None:
            return self.workload

        workload_path = path or DEFAULT_WORKLOAD_PATH
        data = json.loads(workload_path.read_text())
        self.workload = WorkloadConfig(**data)
        return self.workload

    def build_container_env(self) -> dict[str, str]:
        """Build the full set of environment variables for the bench container."""
        env: dict[str, str] = {}

        import uuid as _uuid

        # Unique run ID — used in S3 result key to prevent collisions
        env["GPUSCALE_RUN_ID"] = _uuid.uuid4().hex[:12]

        # Required
        env["MODEL"] = self.model
        env["ENGINE"] = self.engine.value

        # Distributed env for Meta native inference (fairscale requires these)
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29500"

        # Model format / quantization
        if self.model_format:
            env["MODEL_FORMAT"] = self.model_format
        if self.gguf_quant:
            env["GGUF_QUANT"] = self.gguf_quant

        # S3 credentials — only include if we have bucket + keys
        if self.s3.access_key and self.s3.bucket:
            env["AWS_ACCESS_KEY_ID"] = self.s3.access_key
            env["AWS_SECRET_ACCESS_KEY"] = self.s3.secret_key
            env["S3_ENDPOINT"] = self.s3.endpoint
            env["S3_BUCKET"] = self.s3.bucket
            env["AWS_DEFAULT_REGION"] = os.getenv("WASABI_REGION", "us-east-1")

            # Auto-derive model key from model name + format if not explicitly set
            model_key = self.s3.model_key
            if not model_key and self.model:
                fmt = self.model_format or "full"
                model_key = f"{self.model}/{fmt}"
            if model_key:
                env["S3_MODEL_KEY"] = model_key

        # HuggingFace token
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token

        # Workload config as JSON
        if self.workload:
            env["WORKLOAD_CONFIG"] = json.dumps(
                self.workload.model_dump(), separators=(",", ":")
            )

        return env
