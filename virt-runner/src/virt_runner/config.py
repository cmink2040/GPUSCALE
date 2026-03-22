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
DEFAULT_BENCH_IMAGE = "gpuscale-bench:latest"
DEFAULT_S3_ENDPOINT = "https://s3.wasabisys.com"


class ProviderConfig(BaseModel):
    """Provider-specific provisioning configuration."""

    api_key: str = ""
    gpu_type: str = ""  # e.g. "RTX_4090", "A100_80GB"
    gpu_count: int = 1
    disk_gb: int = 40
    image: str = ""  # cloud VM image / template id
    region: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class S3Config(BaseModel):
    """S3/Wasabi configuration for model storage."""

    endpoint: str = DEFAULT_S3_ENDPOINT
    access_key: str = ""
    secret_key: str = ""
    bucket: str = ""
    model_key: str = ""  # object key within bucket, e.g. "models/llama-7b.gguf"

    @model_validator(mode="after")
    def _fill_from_env(self) -> "S3Config":
        if not self.access_key:
            self.access_key = os.getenv("WASABI_ACCESS_KEY", "")
        if not self.secret_key:
            self.secret_key = os.getenv("WASABI_SECRET_KEY", "")
        if not self.bucket:
            self.bucket = os.getenv("WASABI_BUCKET", "")
        return self


class JobConfig(BaseModel):
    """Top-level benchmark job configuration."""

    provider: Provider = Provider.LOCAL
    engine: InferenceEngine = InferenceEngine.LLAMA_CPP
    model: str = ""  # model identifier, e.g. "llama-3-8b-q4"
    bench_image: str = DEFAULT_BENCH_IMAGE
    provider_config: ProviderConfig = Field(default_factory=ProviderConfig)
    s3: S3Config = Field(default_factory=S3Config)
    workload: WorkloadConfig | None = None
    ssh_key_path: str = ""
    ssh_user: str = "root"
    timeout_s: int = 1800  # 30 min default

    @model_validator(mode="after")
    def _fill_api_keys(self) -> "JobConfig":
        """Populate API keys from environment variables if not set explicitly."""
        if not self.provider_config.api_key:
            env_map = {
                Provider.VAST: "VAST_API_KEY",
                Provider.RUNPOD: "RUNPOD_API_KEY",
            }
            env_var = env_map.get(self.provider, "")
            if env_var:
                self.provider_config.api_key = os.getenv(env_var, "")
        return self

    def load_workload(self, path: Path | None = None) -> WorkloadConfig:
        """Load workload from a JSON file, falling back to the bundled default."""
        if self.workload is not None:
            return self.workload

        workload_path = path or DEFAULT_WORKLOAD_PATH
        data = json.loads(workload_path.read_text())
        self.workload = WorkloadConfig(**data)
        return self.workload
