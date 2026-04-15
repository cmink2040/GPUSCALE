"""CLI entrypoint for dbops — submit, list, flag, delete, migrate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_env_path = Path(__file__).resolve()
for _parent in _env_path.parents:
    _candidate = _parent / ".env"
    if _candidate.exists():
        load_dotenv(_candidate)
        break

import click
from rich.console import Console
from rich.table import Table

from dbops import db, validate
from dbops.models import Engine as EngineEnum
from dbops.models import GpuPriceCreate, PriceSource, PriceUnit, Provider
from dbops.pricing import (
    EBAY_TARGETS,
    collect_all_sources,
    fetch_ebay_candidates,
    normalize_gpu_name,
)

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


# ---------------------------------------------------------------------------
# gpu-price group
# ---------------------------------------------------------------------------


@cli.group("gpu-price")
def gpu_price() -> None:
    """Manage GPU price snapshots (gpu_prices table).

    Two flavors of source:

    \b
    - Hardware buy-once prices (ebay, amazon): use `gpu-price add`.
      * ebay:   lowest listing from a seller with 5K+ positive reviews
                where the listing matches the exact GPU SKU
      * amazon: lowest "with reviews" listing for the exact SKU

    \b
    - Cloud hourly rentals (vast, vast (community), runpod): use
      `gpu-price collect-cloud`. Reads current offers from the
      provider APIs and inserts one row per GPU type per source with
      the lowest dph observed.
    """


def _pick_unit(source: PriceSource) -> PriceUnit:
    if source in (PriceSource.EBAY, PriceSource.AMAZON):
        return PriceUnit.ONE_TIME
    return PriceUnit.PER_HOUR


@gpu_price.command("add")
@click.option("--gpu", "gpu_name", required=True, help="GPU name (e.g. 'RTX 4090').")
@click.option("--vram-gb", type=float, required=True, help="VRAM in GB (e.g. 24).")
@click.option(
    "--source",
    type=click.Choice([s.value for s in PriceSource]),
    required=True,
    help="Pricing source.",
)
@click.option("--price", type=float, required=True, help="Price in USD.")
@click.option("--url", "listing_url", default=None, help="Listing URL (ebay/amazon).")
@click.option("--seller", default=None, help="Seller name (ebay).")
@click.option("--notes", default=None, help="Freeform notes.")
def gp_add(
    gpu_name: str,
    vram_gb: float,
    source: str,
    price: float,
    listing_url: str | None,
    seller: str | None,
    notes: str | None,
) -> None:
    """Add a GPU price snapshot (manual entry, typically for ebay/amazon)."""
    src = PriceSource(source)
    unit = _pick_unit(src)
    normalized = normalize_gpu_name(gpu_name)
    payload = GpuPriceCreate(
        gpu_name=normalized,
        gpu_vram_gb=vram_gb,
        source=src,
        unit=unit,
        price_usd=price,
        listing_url=listing_url,
        seller=seller,
        notes=notes,
    )
    with db.get_session() as session:
        row = db.insert_price(session, payload.to_orm())
        console.print(
            f"[green]Inserted[/green] {row.gpu_name} ({row.gpu_vram_gb} GB) "
            f"{row.source} ${row.price_usd:.4f} [dim]id={row.id}[/dim]"
        )


@gpu_price.command("list")
@click.option("--gpu", default=None, help="Filter by GPU name substring.")
@click.option(
    "--source",
    default=None,
    type=click.Choice([s.value for s in PriceSource]),
    help="Filter by source.",
)
@click.option("-n", "--limit", default=50, show_default=True)
def gp_list(gpu: str | None, source: str | None, limit: int) -> None:
    """List GPU price rows, newest first."""
    # Snapshot rows into plain tuples inside the session so display can
    # happen after it closes (SQLAlchemy lazy-loads by default).
    with db.get_session() as session:
        rows = db.list_prices(session, limit=limit, gpu_name=gpu, source=source)
        snapshots = [
            (
                r.gpu_name,
                float(r.gpu_vram_gb),
                r.source,
                float(r.price_usd),
                r.unit,
                str(r.collected_at)[:19] if r.collected_at else "",
            )
            for r in rows
        ]

    if not snapshots:
        console.print("[dim]No prices found.[/dim]")
        return

    table = Table(title=f"GPU prices ({len(snapshots)})", show_lines=False)
    table.add_column("GPU")
    table.add_column("VRAM", justify="right")
    table.add_column("Source")
    table.add_column("Price (USD)", justify="right")
    table.add_column("Unit")
    table.add_column("Collected", style="dim")
    for gpu_name, vram, src, price, unit, collected in snapshots:
        price_display = f"${price:.4f}" if unit == "per_hour" else f"${price:.2f}"
        table.add_row(
            gpu_name,
            f"{vram:.0f}",
            src,
            price_display,
            unit,
            collected,
        )
    console.print(table)


@gpu_price.command("latest")
@click.option("--gpu", default=None, help="Filter by GPU name substring.")
def gp_latest(gpu: str | None) -> None:
    """Show the most recent price per (GPU, source)."""
    with db.get_session() as session:
        rows = db.latest_prices(session, gpu_name=gpu)
        # Snapshot into plain dicts so we can close the session cleanly.
        snapshots = [
            {
                "gpu_name": r.gpu_name,
                "gpu_vram_gb": float(r.gpu_vram_gb),
                "source": r.source,
                "price_usd": float(r.price_usd),
            }
            for r in rows
        ]

    if not snapshots:
        console.print("[dim]No prices found.[/dim]")
        return

    # Group by gpu_name
    by_gpu: dict[str, list[dict]] = {}
    for r in snapshots:
        by_gpu.setdefault(r["gpu_name"], []).append(r)

    table = Table(title=f"Latest GPU prices ({len(by_gpu)} GPUs, {len(snapshots)} snapshots)")
    table.add_column("GPU")
    table.add_column("VRAM", justify="right")
    table.add_column("ebay", justify="right")
    table.add_column("amazon", justify="right")
    table.add_column("vast/hr", justify="right")
    table.add_column("vast-comm/hr", justify="right")
    table.add_column("runpod/hr", justify="right")

    def fmt(price: float | None, unit: str) -> str:
        if price is None:
            return "[dim]-[/dim]"
        return f"${price:.2f}" if unit == "one_time" else f"${price:.4f}"

    for gpu_name in sorted(by_gpu.keys()):
        group = {r["source"]: r["price_usd"] for r in by_gpu[gpu_name]}
        vram = by_gpu[gpu_name][0]["gpu_vram_gb"]
        table.add_row(
            gpu_name,
            f"{vram:.0f}",
            fmt(group.get("ebay"), "one_time"),
            fmt(group.get("amazon"), "one_time"),
            fmt(group.get("vast"), "per_hour"),
            fmt(group.get("vast (community)"), "per_hour"),
            fmt(group.get("runpod"), "per_hour"),
        )
    console.print(table)


@gpu_price.command("collect-cloud")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be inserted without touching the DB.",
)
def gp_collect_cloud(dry_run: bool) -> None:
    """Run every automated cloud collector and insert a snapshot per GPU.

    Sources:

    \b
    - vast              vast.ai datacenter offers (min dph per GPU type)
    - vast (community)  vast.ai community offers (min dph per GPU type)
    - runpod            RunPod gpuTypes.lowestPrice (secure + community)

    Auth:

    \b
    - VAST_API_KEY
    - RUNPOD_API_KEY

    A failing source doesn't block the others. eBay is NOT part of this
    command — see `gpu-price ebay-candidates` for the human-in-the-loop
    eBay flow.
    """
    console.print("[bold]Collecting cloud prices…[/bold]")
    results = collect_all_sources()

    for key in list(results.keys()):
        if key.endswith("_error"):
            source = key[: -len("_error")]
            console.print(f"[red]{source}: {results[key][0]}[/red]")

    total_rows = 0
    for source_label in ("vast", "vast (community)", "runpod"):
        rows = [
            r for r in results.get(source_label, []) if not isinstance(r, Exception)
        ]
        console.print(f"\n[bold]{source_label}[/bold]: {len(rows)} GPU(s)")
        for r in sorted(rows, key=lambda x: x.price_usd):
            console.print(
                f"  {r.gpu_name:24s} {r.gpu_vram_gb:3.0f} GB   "
                f"[green]${r.price_usd:.4f}/hr[/green]"
            )
        total_rows += len(rows)

    if dry_run:
        console.print(f"\n[yellow]Dry run — {total_rows} rows not inserted.[/yellow]")
        return

    console.print(f"\n[bold]Inserting {total_rows} rows…[/bold]")
    inserted = 0
    with db.get_session() as session:
        for source_label in ("vast", "vast (community)", "runpod"):
            rows = [
                r for r in results.get(source_label, []) if not isinstance(r, Exception)
            ]
            for payload in rows:
                db.insert_price(session, payload.to_orm())
                inserted += 1
    console.print(f"[green]Inserted {inserted} price rows.[/green]")


@gpu_price.command("ebay-candidates")
@click.option("--gpu", "gpu_name", required=True, help="Target GPU name (e.g. 'RTX 4090').")
@click.option("--vram", "vram_gb", type=float, default=None, help="VRAM in GB. Appended to the search query to disambiguate variants (e.g. V100 16 vs 32).")
@click.option("--query", default=None, help="Override the search query. Default: 'Nvidia <gpu> <vram>GB'.")
@click.option("-n", "--limit", default=10, show_default=True, help="Max candidates to return.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a human-readable table.")
@click.option("--no-description", is_flag=True, help="Skip the getItem description fetch (faster but less useful for AI review).")
def gp_ebay_candidates(
    gpu_name: str,
    vram_gb: float | None,
    query: str | None,
    limit: int,
    as_json: bool,
    no_description: bool,
) -> None:
    """Fetch eBay candidate listings for AI review (does NOT auto-pick).

    This is the external-agent flow for eBay pricing. The tool:

    \b
      1. Mints an eBay OAuth token (client credentials grant)
      2. Searches `item_summary/search` sorted cheapest-first
      3. Pre-filters by seller.feedbackScore >= 5000 (noise floor only)
      4. Fetches the full description for each kept listing via getItem
      5. Prints the candidates — including price, shipping, total, seller
         feedback, title, condition, listing URL, and the description
         text — and stops

    An external AI agent (e.g. another Claude Code instance) reads the
    output, decides which listing is legitimate (catching parts-only
    traps, 'READ DESCRIPTION' gotchas, SKU mismatches, suspiciously low
    prices, etc.), and then calls `dbops gpu-price add --source ebay ...`
    to record its choice. No auto-picking is ever done by this tool.

    Requires EBAY_CLIENT_ID and EBAY_CLIENT_SECRET env vars.
    """
    try:
        result = fetch_ebay_candidates(
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            query=query,
            limit=limit,
            with_description=not no_description,
        )
    except Exception as exc:
        error_console.print(f"[red]ebay fetch failed: {exc}[/red]")
        raise click.exceptions.Exit(1)

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, indent=2, default=str))
        return

    # Human-readable output — table of candidates followed by descriptions
    console.print()
    console.print(
        f"[bold]eBay candidates[/bold] for [cyan]{result['gpu_name']}[/cyan]"
        + (f" {result['vram_gb']:.0f} GB" if result["vram_gb"] else "")
        + f"  [dim](query: {result['query']})[/dim]"
    )
    console.print(
        f"[dim]Fetched {result['fetched_at']} · "
        f"seller feedback floor >= {result['seller_feedback_floor']} · "
        f"{len(result['candidates'])} candidates[/dim]"
    )

    if not result["candidates"]:
        console.print("[yellow]No candidates passed the feedback floor.[/yellow]")
        return

    table = Table(show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Price", justify="right")
    table.add_column("Ship", justify="right", style="dim")
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Title")
    table.add_column("Seller")
    table.add_column("Feedback", justify="right")
    for c in result["candidates"]:
        ship = c.get("shipping_usd")
        ship_str = f"${ship:.2f}" if ship is not None else "—"
        fb = c["seller"].get("feedback_score")
        pct = c["seller"].get("feedback_percentage")
        fb_str = f"{fb:,} ({pct}%)" if fb is not None else "—"
        table.add_row(
            str(c["rank"]),
            f"${c['price_usd']:.2f}",
            ship_str,
            f"${c['total_usd']:.2f}",
            (c.get("title") or "")[:60],
            (c["seller"].get("username") or "")[:20],
            fb_str,
        )
    console.print(table)

    # Separately dump each description for the agent to read
    console.print()
    console.print("[bold]Descriptions[/bold] [dim](read carefully — this is where the scams live)[/dim]")
    for c in result["candidates"]:
        desc = c.get("description") or "[no description available]"
        console.print()
        console.print(
            f"[bold cyan]#{c['rank']}[/bold cyan]  "
            f"${c['price_usd']:.2f}  "
            f"[dim]{c.get('item_id', '')}[/dim]"
        )
        console.print(f"  [bold]{c.get('title')}[/bold]")
        console.print(f"  [dim]{c.get('listing_url')}[/dim]")
        # Indent description
        for line in desc.split("\n"):
            console.print(f"  {line}")

    console.print()
    console.print(
        "[dim]To record a chosen listing: [/dim]"
        f"[bold]dbops gpu-price add --gpu '{result['gpu_name']}' "
        f"--vram-gb {result['vram_gb']:.0f} --source ebay "
        "--price <USD> --url '<listing_url>' --seller '<username>' "
        "--notes '<why this one>'[/bold]"
    )


@gpu_price.command("ebay-targets")
def gp_ebay_targets() -> None:
    """Print the default list of GPUs that an agent might want to review.

    These are the GPUs we currently benchmark — a good iteration list for
    an agent that wants to refresh eBay prices for everything in one pass.
    """
    table = Table(title=f"Default eBay target list ({len(EBAY_TARGETS)} GPUs)")
    table.add_column("GPU")
    table.add_column("VRAM", justify="right")
    table.add_column("Default query", style="dim")
    for gpu_name, vram, query in EBAY_TARGETS:
        table.add_row(gpu_name, f"{int(vram)} GB", query)
    console.print(table)


if __name__ == "__main__":
    cli()
