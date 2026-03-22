"""CLI entrypoint for dbops — submit, list, flag, delete, init-db."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from dbops import db, validate
from dbops.models import Engine, Provider

console = Console()
error_console = Console(stderr=True)

SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


@click.group()
@click.version_option(package_name="gpuscale-dbops")
def cli() -> None:
    """GPUSCALE dbops — manage GPU benchmark results in Supabase."""


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("json_file", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Validate only, do not insert.")
def submit(json_file: Path, dry_run: bool) -> None:
    """Submit a benchmark result from a JSON file."""
    # Load
    data, load_err = validate.load_json_file(json_file)
    if load_err:
        error_console.print(f"[red]Error:[/red] {load_err}")
        sys.exit(1)

    assert data is not None

    # Validate
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

    # Insert
    client = db.get_client()
    row = db.insert_result(client, result.to_insert_dict())
    console.print(f"[green]Inserted:[/green] id={row['id']}")


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
    type=click.Choice([e.value for e in Engine], case_sensitive=False),
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
    client = db.get_client()
    rows = db.list_results(
        client,
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
        rid = str(r["id"])[:8]
        flagged_str = "[red]YES[/red]" if r.get("flagged") else ""
        created = str(r.get("created_at", ""))[:19]
        table.add_row(
            rid,
            r.get("gpu_name", ""),
            r.get("model_name", ""),
            r.get("engine", ""),
            r.get("quantization", ""),
            r.get("provider", ""),
            f"{r.get('tokens_per_sec', 0):.1f}",
            f"{r.get('time_to_first_token_ms', 0):.1f}",
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
    client = db.get_client()
    row = db.flag_result(client, result_id)
    if row:
        console.print(f"[yellow]Flagged:[/yellow] {row['id']}")
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

    client = db.get_client()
    row = db.delete_result(client, result_id)
    if row:
        console.print(f"[red]Deleted:[/red] {row['id']}")
    else:
        error_console.print(f"[red]Not found:[/red] {result_id}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# init-db
# ---------------------------------------------------------------------------
@cli.command("init-db")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Write SQL to file.")
def init_db(output: Optional[Path]) -> None:
    """Print or save the SQL schema for the benchmark_results table."""
    sql = SCHEMA_SQL_PATH.read_text()

    if output:
        output.write_text(sql)
        console.print(f"[green]Schema written to:[/green] {output}")
    else:
        console.print(sql)


if __name__ == "__main__":
    cli()
