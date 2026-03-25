"""Load and validate the models.toml configuration."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "models.toml"


@dataclass(frozen=True)
class FormatSpec:
    """A single format entry for a model."""

    type: str  # "full", "gguf", or "gptq"
    storage: str = "huggingface"  # "huggingface" (pull direct) or "s3" (pull from Wasabi)
    quants: list[str] = field(default_factory=list)
    variant: str = ""
    gguf_repo_id: str = ""
    gptq_repo_id: str = ""

    def __post_init__(self) -> None:
        valid_types = {"full", "gguf", "gptq"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid format type '{self.type}'. Must be one of {valid_types}")
        valid_storage = {"huggingface", "s3"}
        if self.storage not in valid_storage:
            raise ValueError(f"Invalid storage '{self.storage}'. Must be one of {valid_storage}")
        if self.type == "gguf" and not self.quants:
            raise ValueError("GGUF format requires at least one entry in 'quants'")
        if self.type == "gptq" and not self.variant:
            raise ValueError("GPTQ format requires a 'variant' (e.g. '4bit-128g')")

    @property
    def is_s3(self) -> bool:
        """Whether this format is stored on S3 (private/gated/meta)."""
        return self.storage == "s3"


@dataclass(frozen=True)
class ModelSpec:
    """A single model entry from models.toml."""

    repo_id: str
    formats: list[FormatSpec]
    source: str = "huggingface"  # "huggingface" or "meta"
    meta_model_id: str = ""  # e.g. "llama3.1-8b-instruct" (required when source="meta")

    def __post_init__(self) -> None:
        valid_sources = {"huggingface", "meta"}
        if self.source not in valid_sources:
            raise ValueError(f"Invalid source '{self.source}'. Must be one of {valid_sources}")
        if self.source == "meta" and not self.meta_model_id:
            raise ValueError(
                f"Model '{self.repo_id}' uses source='meta' but no 'meta_model_id' was provided. "
                "Set meta_model_id to the Meta CLI model ID (e.g. 'llama3.1-8b-instruct')."
            )

    @property
    def org(self) -> str:
        """Organization / namespace portion of repo_id (e.g. 'meta-llama')."""
        return self.repo_id.split("/")[0]

    @property
    def name(self) -> str:
        """Model name portion of repo_id (e.g. 'Llama-3.1-8B-Instruct')."""
        return self.repo_id.split("/")[1]


def load_config(path: Path | None = None) -> list[ModelSpec]:
    """Load and validate models.toml, returning a list of ModelSpec objects.

    Args:
        path: Path to the TOML config file. Defaults to models.toml next to the package root.

    Returns:
        List of validated ModelSpec objects.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is malformed.
    """
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    log.info("Loading config from %s", config_path)
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    models_raw = raw.get("models")
    if not models_raw:
        raise ValueError("Config must contain at least one [[models]] entry")

    models: list[ModelSpec] = []
    for idx, entry in enumerate(models_raw):
        repo_id = entry.get("repo_id")
        if not repo_id or "/" not in repo_id:
            raise ValueError(
                f"models[{idx}]: 'repo_id' must be in 'org/name' format, got '{repo_id}'"
            )

        formats_raw = entry.get("formats")
        if not formats_raw:
            raise ValueError(f"models[{idx}] ({repo_id}): must have at least one [[models.formats]]")

        formats: list[FormatSpec] = []
        for fmt in formats_raw:
            fmt_type = fmt.get("type")
            if not fmt_type:
                raise ValueError(f"models[{idx}] ({repo_id}): each format needs a 'type' field")
            formats.append(
                FormatSpec(
                    type=fmt_type,
                    storage=fmt.get("storage", "huggingface"),
                    quants=fmt.get("quants", []),
                    variant=fmt.get("variant", ""),
                    gguf_repo_id=fmt.get("gguf_repo_id", ""),
                    gptq_repo_id=fmt.get("gptq_repo_id", ""),
                )
            )

        source = entry.get("source", "huggingface")
        meta_model_id = entry.get("meta_model_id", "")

        models.append(ModelSpec(
            repo_id=repo_id,
            formats=formats,
            source=source,
            meta_model_id=meta_model_id,
        ))

    log.info("Loaded %d model(s) from config", len(models))
    return models
