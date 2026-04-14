"""SQLAlchemy database engine, session, and query helpers."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from dbops.models import BenchmarkResult, GpuPrice

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_URL")
    if not url:
        print(
            "Error: DATABASE_URL or SUPABASE_URL environment variable is not set.",
            file=sys.stderr,
        )
        print(
            "Set it to your Supabase Postgres connection string, e.g.:",
            file=sys.stderr,
        )
        print(
            "  postgresql://postgres.<ref>:<password>@<host>:5432/postgres",
            file=sys.stderr,
        )
        sys.exit(1)
    return url


def get_engine() -> Engine:
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(_get_database_url(), echo=False, pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional session scope."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def insert_result(session: Session, result: BenchmarkResult) -> BenchmarkResult:
    """Insert a benchmark result. Returns the instance with id/created_at populated."""
    session.add(result)
    session.flush()
    session.refresh(result)
    return result


def list_results(
    session: Session,
    *,
    limit: int = 25,
    gpu_name: str | None = None,
    model_name: str | None = None,
    engine: str | None = None,
    provider: str | None = None,
    quantization: str | None = None,
) -> list[BenchmarkResult]:
    """Query benchmark results with optional filters, ordered by newest first."""
    stmt = select(BenchmarkResult)

    if gpu_name:
        stmt = stmt.where(BenchmarkResult.gpu_name.ilike(f"%{gpu_name}%"))
    if model_name:
        stmt = stmt.where(BenchmarkResult.model_name.ilike(f"%{model_name}%"))
    if engine:
        stmt = stmt.where(BenchmarkResult.engine == engine)
    if provider:
        stmt = stmt.where(BenchmarkResult.provider == provider)
    if quantization:
        stmt = stmt.where(BenchmarkResult.quantization.ilike(f"%{quantization}%"))

    stmt = stmt.order_by(desc(BenchmarkResult.created_at)).limit(limit)
    return list(session.scalars(stmt).all())


def flag_result(session: Session, result_id: str) -> BenchmarkResult | None:
    """Flag a result as suspicious. Returns the updated row or None."""
    result = session.get(BenchmarkResult, result_id)
    if result is None:
        return None
    result.flagged = True
    session.flush()
    return result


def delete_result(session: Session, result_id: str) -> BenchmarkResult | None:
    """Delete a result by ID. Returns the deleted row or None."""
    result = session.get(BenchmarkResult, result_id)
    if result is None:
        return None
    session.delete(result)
    session.flush()
    return result


# ---------------------------------------------------------------------------
# gpu_prices helpers
# ---------------------------------------------------------------------------


def insert_price(session: Session, price: GpuPrice) -> GpuPrice:
    """Insert a GPU price row. Returns the instance with id/collected_at populated."""
    session.add(price)
    session.flush()
    session.refresh(price)
    return price


def list_prices(
    session: Session,
    *,
    limit: int = 100,
    gpu_name: str | None = None,
    source: str | None = None,
) -> list[GpuPrice]:
    """Query GPU prices with optional filters, newest first."""
    stmt = select(GpuPrice)
    if gpu_name:
        stmt = stmt.where(GpuPrice.gpu_name.ilike(f"%{gpu_name}%"))
    if source:
        stmt = stmt.where(GpuPrice.source == source)
    stmt = stmt.order_by(desc(GpuPrice.collected_at)).limit(limit)
    return list(session.scalars(stmt).all())


def latest_prices(session: Session, gpu_name: str | None = None) -> list[GpuPrice]:
    """Return the most recent price per (gpu_name, source). Optional gpu filter.

    Implemented in Python rather than SQL DISTINCT ON so the helper stays
    portable across drivers and works against a small price catalog. The
    expected row count is in the low thousands.
    """
    stmt = select(GpuPrice)
    if gpu_name:
        stmt = stmt.where(GpuPrice.gpu_name.ilike(f"%{gpu_name}%"))
    stmt = stmt.order_by(desc(GpuPrice.collected_at))
    rows = list(session.scalars(stmt).all())
    seen: set[tuple[str, str]] = set()
    latest: list[GpuPrice] = []
    for r in rows:
        key = (r.gpu_name, r.source)
        if key in seen:
            continue
        seen.add(key)
        latest.append(r)
    return latest
