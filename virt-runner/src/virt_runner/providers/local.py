"""Local provider: no provisioning, runs benchmarks on the local machine."""

from __future__ import annotations

from virt_runner.config import JobConfig
from virt_runner.host_info import collect_host_metadata, detect_local_gpus
from virt_runner.providers.base import BaseProvider, ProvisionedInstance


class LocalProvider(BaseProvider):
    """Runs benchmarks directly on the local machine with local GPUs."""

    def __init__(self, config: JobConfig) -> None:
        super().__init__(config)

    def provision(self) -> ProvisionedInstance:
        """Detect local GPUs and host metadata; no actual provisioning needed."""
        gpus = detect_local_gpus()
        host_meta = collect_host_metadata()
        return ProvisionedInstance(
            instance_id="local",
            host="localhost",
            port=0,
            gpus=gpus,
            host_metadata=host_meta,
            is_local=True,
        )

    def wait_ready(self, instance: ProvisionedInstance, timeout_s: int = 300) -> bool:
        """Local instance is always ready."""
        return True

    def teardown(self, instance: ProvisionedInstance) -> None:
        """No-op for local provider."""
        pass

    def get_name(self) -> str:
        return "local"
