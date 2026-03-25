"""CLI entrypoint for s3-attach."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Walk up to find the root .env file
_env_path = Path(__file__).resolve()
for _parent in _env_path.parents:
    _candidate = _parent / ".env"
    if _candidate.exists():
        load_dotenv(_candidate)
        break

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.tree import Tree

from s3_attach.cleanup import cleanup_download_dir, remove_downloaded
from s3_attach.config import FormatSpec, ModelSpec, load_config
from s3_attach.downloader import download_model
from s3_attach.manifest import fetch_manifest, regenerate_manifest
from s3_attach.uploader import get_s3_client, list_bucket_objects, upload_directory

console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # Quiet noisy libraries unless in debug mode
    if not verbose:
        logging.getLogger("boto3").setLevel(logging.WARNING)
        logging.getLogger("botocore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("s3transfer").setLevel(logging.WARNING)


def _s3_prefix_for(model: ModelSpec, fmt: FormatSpec) -> str:
    """Compute the S3 key prefix for a model + format combination.

    Examples:
        meta-llama/Llama-3.1-8B-Instruct + full   -> meta-llama/Llama-3.1-8B-Instruct/full
        meta-llama/Llama-3.1-8B-Instruct + gguf    -> meta-llama/Llama-3.1-8B-Instruct/gguf
        meta-llama/Llama-3.1-8B-Instruct + gptq/4bit-128g
            -> meta-llama/Llama-3.1-8B-Instruct/gptq/4bit-128g
    """
    base = f"{model.org}/{model.name}/{fmt.type}"
    if fmt.type == "gptq" and fmt.variant:
        base = f"{base}/{fmt.variant}"
    return base


@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """s3-attach: Model pool manager for GPUSCALE.

    Downloads models from HuggingFace Hub, uploads them to Wasabi S3,
    and maintains a manifest of available models.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to models.toml config file.",
)
@click.option(
    "--download-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=Path("downloads"),
    show_default=True,
    help="Local directory for staging downloads.",
)
@click.option(
    "--skip-download",
    is_flag=True,
    default=False,
    help="Skip download step (upload previously downloaded files).",
)
@click.option(
    "--skip-upload",
    is_flag=True,
    default=False,
    help="Skip upload step (download only, keep local files).",
)
@click.option(
    "--keep-local",
    is_flag=True,
    default=False,
    help="Do not delete local files after upload.",
)
@click.option(
    "--model",
    "-m",
    "model_filter",
    type=str,
    default=None,
    help="Only sync a specific model (match by repo_id substring, e.g. 'Llama-3.1-8B').",
)
@click.option(
    "--meta-url",
    type=str,
    default=None,
    help="Meta signed download URL for Llama models (or set META_LLAMA_URL env var).",
)
def sync(
    config_path: Path | None,
    download_dir: Path,
    skip_download: bool,
    skip_upload: bool,
    keep_local: bool,
    model_filter: str | None,
    meta_url: str | None,
) -> None:
    """Download models from HuggingFace and upload to Wasabi S3.

    Reads models.toml for the list of models and formats to sync.
    After successful upload, local downloads are cleaned up (unless --keep-local).
    The bucket manifest is regenerated after all uploads complete.
    """
    log = logging.getLogger("s3_attach.sync")

    try:
        models = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    if model_filter:
        models = [m for m in models if model_filter.lower() in m.repo_id.lower()]
        if not models:
            console.print(f"[yellow]No models matched filter:[/yellow] {model_filter}")
            sys.exit(0)

    console.print(f"\n[bold]Syncing {len(models)} model(s)[/bold]\n")

    s3_client = None
    if not skip_upload:
        try:
            s3_client = get_s3_client()
        except EnvironmentError as e:
            console.print(f"[red]S3 configuration error:[/red] {e}")
            sys.exit(1)

    success_count = 0
    error_count = 0

    for model in models:
        console.rule(f"[bold cyan]{model.repo_id}[/bold cyan]")

        for fmt in model.formats:
            fmt_label = fmt.type
            if fmt.type == "gguf":
                fmt_label = f"gguf ({', '.join(fmt.quants)})"
            elif fmt.type == "gptq":
                fmt_label = f"gptq/{fmt.variant}"

            if not fmt.is_s3:
                console.print(f"  Format: [dim]{fmt_label}[/dim] [dim](huggingface — skipped, pulled direct by runner)[/dim]")
                continue

            console.print(f"  Format: [green]{fmt_label}[/green] [cyan](s3)[/cyan]")

            local_path: Path | None = None

            # --- Download ---
            if not skip_download:
                try:
                    local_path = download_model(model, fmt, download_dir, meta_url=meta_url)
                    console.print(f"  Downloaded to: {local_path}")
                except NotImplementedError as e:
                    console.print(f"  [yellow]Skipped (not implemented):[/yellow] {e}")
                    continue
                except Exception as e:
                    console.print(f"  [red]Download failed:[/red] {e}")
                    log.exception("Download error for %s %s", model.repo_id, fmt.type)
                    error_count += 1
                    continue
            else:
                # When skipping download, reconstruct expected local path
                local_path = download_dir / model.org / model.name / fmt.type
                if fmt.type == "gptq" and fmt.variant:
                    local_path = download_dir / model.org / model.name / "gptq" / fmt.variant
                if not local_path.exists():
                    console.print(f"  [yellow]No local files found at {local_path}[/yellow]")
                    continue

            # --- Upload ---
            if not skip_upload and local_path is not None:
                try:
                    s3_prefix = _s3_prefix_for(model, fmt)
                    results = upload_directory(
                        local_path, s3_prefix, s3_client=s3_client
                    )
                    uploaded = sum(1 for r in results if not r.get("skipped"))
                    skipped = sum(1 for r in results if r.get("skipped"))
                    console.print(
                        f"  Uploaded: {uploaded} file(s), skipped: {skipped} (already exist)"
                    )
                    success_count += 1
                except Exception as e:
                    console.print(f"  [red]Upload failed:[/red] {e}")
                    log.exception("Upload error for %s %s", model.repo_id, fmt.type)
                    error_count += 1
                    continue

            # --- Cleanup ---
            if not keep_local and not skip_upload and local_path is not None:
                remove_downloaded(local_path)

    # Clean up the top-level download directory if it is now empty
    if not keep_local and not skip_upload:
        cleanup_download_dir(download_dir)

    # --- Regenerate manifest ---
    if not skip_upload:
        console.print("\n[bold]Regenerating manifest...[/bold]")
        try:
            manifest = regenerate_manifest(s3_client=s3_client)
            model_count = len(manifest.get("models", {}))
            console.print(f"[green]Manifest updated:[/green] {model_count} model(s) indexed")
        except Exception as e:
            console.print(f"[red]Manifest generation failed:[/red] {e}")
            log.exception("Manifest error")

    # --- Summary ---
    console.print()
    if error_count:
        console.print(
            f"[bold yellow]Done with errors:[/bold yellow] "
            f"{success_count} succeeded, {error_count} failed"
        )
        sys.exit(1)
    else:
        console.print(f"[bold green]Done:[/bold green] {success_count} format(s) synced successfully")


@cli.command("list")
@click.option(
    "--prefix",
    "-p",
    default="",
    help="Filter by S3 key prefix (e.g. 'meta-llama/').",
)
def list_models(prefix: str) -> None:
    """List models and files available in the Wasabi S3 bucket."""
    try:
        s3_client = get_s3_client()
    except EnvironmentError as e:
        console.print(f"[red]S3 configuration error:[/red] {e}")
        sys.exit(1)

    objects = list_bucket_objects(prefix=prefix, s3_client=s3_client)

    if not objects:
        console.print("[yellow]No objects found in bucket.[/yellow]")
        return

    # Build a tree view grouped by org/model/format
    tree = Tree(f"[bold]s3://{_get_bucket_display()}[/bold]")
    grouped: dict[str, dict[str, dict[str, list]]] = {}

    for obj in objects:
        parts = obj["key"].split("/")
        if len(parts) < 2:
            continue
        org = parts[0]
        model_name = parts[1] if len(parts) > 1 else "(root)"
        fmt_type = parts[2] if len(parts) > 2 else "(files)"

        grouped.setdefault(org, {}).setdefault(model_name, {}).setdefault(fmt_type, []).append(obj)

    total_size = 0
    total_files = 0

    for org, models in sorted(grouped.items()):
        org_branch = tree.add(f"[cyan]{org}/[/cyan]")
        for model_name, formats in sorted(models.items()):
            model_branch = org_branch.add(f"[blue]{model_name}/[/blue]")
            for fmt_type, files in sorted(formats.items()):
                fmt_size = sum(f["size"] for f in files)
                total_size += fmt_size
                total_files += len(files)
                fmt_branch = model_branch.add(
                    f"[green]{fmt_type}/[/green]  "
                    f"({len(files)} files, {_human_size(fmt_size)})"
                )
                for f in sorted(files, key=lambda x: x["key"]):
                    filename = f["key"].split("/")[-1]
                    fmt_branch.add(f"{filename}  [dim]{_human_size(f['size'])}[/dim]")

    console.print(tree)
    console.print(
        f"\n[bold]Total:[/bold] {total_files} files, {_human_size(total_size)}"
    )


@cli.command()
def manifest() -> None:
    """Regenerate manifest.json from current bucket contents."""
    try:
        s3_client = get_s3_client()
    except EnvironmentError as e:
        console.print(f"[red]S3 configuration error:[/red] {e}")
        sys.exit(1)

    console.print("[bold]Scanning bucket and regenerating manifest...[/bold]")

    try:
        result = regenerate_manifest(s3_client=s3_client)
    except Exception as e:
        console.print(f"[red]Failed:[/red] {e}")
        sys.exit(1)

    model_count = len(result.get("models", {}))
    console.print(f"[green]Manifest regenerated:[/green] {model_count} model(s) indexed")
    console.print(f"Generated at: {result.get('generated_at', 'unknown')}")

    # Show summary table
    models = result.get("models", {})
    if models:
        table = Table(title="Models in manifest")
        table.add_column("Model", style="cyan")
        table.add_column("Formats", style="green")
        table.add_column("Total files", justify="right")

        for repo_id, info in sorted(models.items()):
            formats = list(info.get("formats", {}).keys())
            file_count = 0
            for fmt_key, fmt_data in info.get("formats", {}).items():
                if isinstance(fmt_data, dict) and "files" in fmt_data:
                    file_count += len(fmt_data["files"])
                elif isinstance(fmt_data, dict):
                    # GPTQ with variants
                    for variant_data in fmt_data.values():
                        if isinstance(variant_data, dict) and "files" in variant_data:
                            file_count += len(variant_data["files"])

            table.add_row(repo_id, ", ".join(formats), str(file_count))

        console.print()
        console.print(table)


def _human_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} PB"


def _get_bucket_display() -> str:
    """Get bucket name for display, with graceful fallback."""
    import os

    return os.environ.get("WASABI_BUCKET", "<WASABI_BUCKET>")


if __name__ == "__main__":
    cli()
