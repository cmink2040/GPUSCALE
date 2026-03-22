"""Abstract base class for GPU instance providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from virt_runner.config import JobConfig
from virt_runner.models import GPUInfo, HostMetadata


@dataclass
class ProvisionedInstance:
    """Represents a provisioned GPU instance (cloud or local)."""

    instance_id: str = ""
    host: str = ""  # IP or hostname for SSH
    port: int = 22
    ssh_user: str = "root"
    ssh_key_path: str = ""
    gpus: list[GPUInfo] = field(default_factory=list)
    host_metadata: HostMetadata = field(default_factory=HostMetadata)
    is_local: bool = False
    extra: dict = field(default_factory=dict)


class BaseProvider(abc.ABC):
    """Abstract interface that all providers must implement."""

    def __init__(self, config: JobConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def provision(self) -> ProvisionedInstance:
        """Provision (or detect) a GPU instance. Returns connection details."""
        ...

    @abc.abstractmethod
    def wait_ready(self, instance: ProvisionedInstance, timeout_s: int = 300) -> bool:
        """Block until the instance is ready for SSH / Docker commands.

        Returns True if ready, False on timeout.
        """
        ...

    @abc.abstractmethod
    def teardown(self, instance: ProvisionedInstance) -> None:
        """Destroy / release the instance. No-op for local provider."""
        ...

    @abc.abstractmethod
    def get_name(self) -> str:
        """Human-readable provider name."""
        ...
