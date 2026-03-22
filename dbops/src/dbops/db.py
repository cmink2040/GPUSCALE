"""Supabase client wrapper — connection init, insert, query, flag, delete."""

from __future__ import annotations

import os
import sys
from typing import Any

from supabase import Client, create_client

TABLE = "benchmark_results"


def _get_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Error: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return value


def get_client() -> Client:
    """Create and return a Supabase client using service-role credentials."""
    url = _get_env("SUPABASE_URL")
    key = _get_env("SUPABASE_SERVICE_KEY")
    return create_client(url, key)


def insert_result(client: Client, data: dict[str, Any]) -> dict[str, Any]:
    """Insert a validated benchmark result. Returns the inserted row."""
    response = client.table(TABLE).insert(data).execute()
    return response.data[0]


def list_results(
    client: Client,
    *,
    limit: int = 25,
    gpu_name: str | None = None,
    model_name: str | None = None,
    engine: str | None = None,
    provider: str | None = None,
    quantization: str | None = None,
) -> list[dict[str, Any]]:
    """Query benchmark results with optional filters, ordered by newest first."""
    query = client.table(TABLE).select("*")

    if gpu_name:
        query = query.ilike("gpu_name", f"%{gpu_name}%")
    if model_name:
        query = query.ilike("model_name", f"%{model_name}%")
    if engine:
        query = query.eq("engine", engine)
    if provider:
        query = query.eq("provider", provider)
    if quantization:
        query = query.ilike("quantization", f"%{quantization}%")

    query = query.order("created_at", desc=True).limit(limit)
    response = query.execute()
    return response.data


def flag_result(client: Client, result_id: str) -> dict[str, Any] | None:
    """Flag a result as suspicious. Returns the updated row or None if not found."""
    response = (
        client.table(TABLE)
        .update({"flagged": True})
        .eq("id", result_id)
        .execute()
    )
    if response.data:
        return response.data[0]
    return None


def delete_result(client: Client, result_id: str) -> dict[str, Any] | None:
    """Delete a result by ID. Returns the deleted row or None if not found."""
    response = (
        client.table(TABLE)
        .delete()
        .eq("id", result_id)
        .execute()
    )
    if response.data:
        return response.data[0]
    return None
