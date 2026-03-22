"""Validation logic for benchmark result submissions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dbops.models import BenchmarkResultCreate, Engine, Provider


# Canonical sets for quick membership checks outside pydantic
VALID_PROVIDERS = {p.value for p in Provider}
VALID_ENGINES = {e.value for e in Engine}


class ValidationReport:
    """Collects validation errors and warnings for a single submission."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def load_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load and parse a JSON file, returning (data, error)."""
    if not path.exists():
        return None, f"File not found: {path}"
    if not path.is_file():
        return None, f"Not a file: {path}"
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "JSON root must be an object"
    return data, None


def validate_result(data: dict[str, Any]) -> tuple[BenchmarkResultCreate | None, ValidationReport]:
    """
    Validate a raw dict against the benchmark result schema.

    Returns the validated model (or None) and a report with any errors/warnings.
    """
    report = ValidationReport()

    # --- Extra semantic checks that go beyond pydantic ---
    provider = data.get("provider")
    engine = data.get("engine")

    # Warn about local-only fields when provider != local
    if provider and provider != "local":
        for field in ("host_os", "host_kernel", "host_driver_version"):
            if data.get(field):
                report.add_warning(
                    f"'{field}' is typically only set for provider='local', "
                    f"but provider='{provider}'"
                )

    # Warn if cloud provider but no container image
    if provider in ("vast.ai", "runpod") and not data.get("container_image"):
        report.add_warning(
            f"provider='{provider}' but no container_image specified"
        )

    # Warn about suspiciously high tok/s
    tps = data.get("tokens_per_sec")
    if isinstance(tps, (int, float)) and tps > 10_000:
        report.add_warning(
            f"tokens_per_sec={tps} is unusually high — double-check the value"
        )

    # Warn about suspiciously low TTFT
    ttft = data.get("time_to_first_token_ms")
    if isinstance(ttft, (int, float)) and ttft < 1:
        report.add_warning(
            f"time_to_first_token_ms={ttft} is suspiciously low"
        )

    # --- Pydantic validation (catches type errors, enums, ranges) ---
    try:
        result = BenchmarkResultCreate(**data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err["loc"])
            report.add_error(f"{loc}: {err['msg']}")
        return None, report

    return result, report
