"""CLI entrypoint for dbops — submit, list, flag, delete, migrate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from dbops import db, validate
from dbops.models import Engine as EngineEnum
from dbops.models import Provider

console = Console()
error_console = Console(stderr=True)

ALEMBIC_DIR = Path(__file__).parent.parent.parent / "alembic"


@click.group()
@click.version_option(package_name="gpuscale-dbops")
def cli() -> None:
    """GPUSCALE dbops — manage GPU benchmark results."""


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("json_file", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Validate only, do not insert.")
def submit(json_file: Path, dry_run: bool) -> None:
    """Submit a benchmark result from a JSON file."""
    data, load_err = validate.load_json_file(json_file)
    if load_err:
        error_console.print(f"[red]Error:[/red] {load_err}")
        sys.exit(1)

    assert data is not None

    result, report = validate.validate_result(data)

    for w in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")

    if not report.ok:
        error_console.print("[red]Validation failed:[/red]")
        for e in report.errors:
            error_console.print(f"  - {e}")
        sys.exit(1)

    assert result is not None
    console.print("[green]Validation passed.[/green]")

    if dry_run:
        console.print("[dim]Dry run — nothing inserted.[/dim]")
        return

    orm_obj = result.to_orm()
    with db.get_session() as session:
        row = db.insert_result(session, orm_obj)
        console.print(f"[green]Inserted:[/green] id={row.id}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
@cli.command("list")
@click.option("-n", "--limit", default=25, show_default=True, help="Max results.")
@click.option("--gpu", default=None, help="Filter by GPU name (substring).")
@click.option("--model", default=None, help="Filter by model name (substring).")
@click.option(
    "--engine",
    default=None,
    type=click.Choice([e.value for e in EngineEnum], case_sensitive=False),
    help="Filter by engine.",
)
@click.option(
    "--provider",
    default=None,
    type=click.Choice([p.value for p in Provider], case_sensitive=False),
    help="Filter by provider.",
)
@click.option("--quant", default=None, help="Filter by quantization (substring).")
def list_results(
    limit: int,
    gpu: Optional[str],
    model: Optional[str],
    engine: Optional[str],
    provider: Optional[str],
    quant: Optional[str],
) -> None:
    """List recent benchmark results with optional filters."""
    with db.get_session() as session:
        rows = db.list_results(
            session,
            limit=limit,
            gpu_name=gpu,
            model_name=model,
            engine=engine,
            provider=provider,
            quantization=quant,
        )

        if not rows:
            console.print("[dim]No results found.[/dim]")
            return

        table = Table(title=f"Benchmark Results ({len(rows)})", show_lines=False)
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("GPU")
        table.add_column("Model")
        table.add_column("Engine")
        table.add_column("Quant")
        table.add_column("Provider")
        table.add_column("tok/s", justify="right")
        table.add_column("TTFT ms", justify="right")
        table.add_column("Flagged")
        table.add_column("Created", style="dim")

        for r in rows:
            rid = str(r.id)[:8]
            flagged_str = "[red]YES[/red]" if r.flagged else ""
            created = str(r.created_at)[:19] if r.created_at else ""
            table.add_row(
                rid,
                r.gpu_name or "",
                r.model_name or "",
                r.engine or "",
                r.quantization or "",
                r.provider or "",
                f"{r.tokens_per_sec:.1f}" if r.tokens_per_sec else "",
                f"{r.time_to_first_token_ms:.1f}" if r.time_to_first_token_ms else "",
                flagged_str,
                created,
            )

        console.print(table)


# ---------------------------------------------------------------------------
# flag
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("result_id")
def flag(result_id: str) -> None:
    """Flag a suspicious benchmark result by ID."""
    with db.get_session() as session:
        row = db.flag_result(session, result_id)
        if row:
            console.print(f"[yellow]Flagged:[/yellow] {row.id}")
        else:
            error_console.print(f"[red]Not found:[/red] {result_id}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("result_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def delete(result_id: str, yes: bool) -> None:
    """Delete a benchmark result by ID (admin)."""
    if not yes:
        click.confirm(f"Delete result {result_id}?", abort=True)

    with db.get_session() as session:
        row = db.delete_result(session, result_id)
        if row:
            console.print(f"[red]Deleted:[/red] {row.id}")
        else:
            error_console.print(f"[red]Not found:[/red] {result_id}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--revision", default="head", help="Target revision (default: head).")
def migrate(revision: str) -> None:
    """Run Alembic migrations to bring the database schema up to date."""
    from alembic import command
    from alembic.config import Config

    alembic_ini = ALEMBIC_DIR.parent / "alembic.ini"
    if not alembic_ini.exists():
        error_console.print(f"[red]Error:[/red] alembic.ini not found at {alembic_ini}")
        sys.exit(1)

    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("script_location", str(ALEMBIC_DIR))

    console.print(f"[dim]Running migrations to revision: {revision}[/dim]")
    command.upgrade(alembic_cfg, revision)
    console.print("[green]Migrations complete.[/green]")


# ---------------------------------------------------------------------------
# revision (create new migration)
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("message")
@click.option("--autogenerate/--no-autogenerate", default=True, help="Auto-detect model changes.")
def revision(message: str, autogenerate: bool) -> None:
    """Create a new Alembic migration revision."""
    from alembic import command
    from alembic.config import Config

    alembic_ini = ALEMBIC_DIR.parent / "alembic.ini"
    if not alembic_ini.exists():
        error_console.print(f"[red]Error:[/red] alembic.ini not found at {alembic_ini}")
        sys.exit(1)

    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("script_location", str(ALEMBIC_DIR))

    console.print(f"[dim]Creating revision: {message}[/dim]")
    command.revision(alembic_cfg, message=message, autogenerate=autogenerate)
    console.print("[green]Revision created.[/green]")


if __name__ == "__main__":
    cli()
