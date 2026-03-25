"""Download models from Meta's official distribution or HuggingFace Hub."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from s3_attach.config import FormatSpec, ModelSpec

log = logging.getLogger(__name__)

DEFAULT_DOWNLOAD_DIR = Path("downloads")


def download_model(
    model: ModelSpec,
    fmt: FormatSpec,
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
    meta_url: str | None = None,
) -> Path:
    """Download a model in the specified format.

    Routes to Meta's official distribution or HuggingFace Hub based on
    the model's source config.

    Args:
        meta_url: Meta signed download URL. Required for source="meta" models.
                  Can also be set via META_LLAMA_URL env var.

    Returns the local directory path containing the downloaded artifacts.
    """
    if fmt.type == "full" and model.source == "meta":
        url = meta_url or os.environ.get("META_LLAMA_URL")
        if not url:
            raise RuntimeError(
                f"Model '{model.repo_id}' requires a Meta signed URL.\n"
                "Get one at https://llama.meta.com/ (accept the license, URL is emailed to you).\n"
                "Then either:\n"
                "  - Pass --meta-url <URL>\n"
                "  - Set META_LLAMA_URL=<URL> in your .env"
            )
        return _download_meta_full(model, download_dir, url)
    elif fmt.type == "full":
        return _download_hf_full(model, download_dir)
    elif fmt.type == "gguf":
        return _download_hf_gguf(model, fmt, download_dir)
    elif fmt.type == "gptq":
        return _download_hf_gptq(model, fmt, download_dir)
    else:
        raise ValueError(f"Unknown format type: {fmt.type}")


# ---------------------------------------------------------------------------
# Meta official distribution (via llama_models Python API)
# ---------------------------------------------------------------------------


def _download_meta_full(model: ModelSpec, download_dir: Path, signed_url: str) -> Path:
    """Download full weights from Meta using the llama_models library.

    Uses the llama_models Python API directly to resolve the model's file
    manifest, then calls its parallel downloader. This is the same logic
    the `llama-model download` CLI uses internally.
    """
    import asyncio

    from llama_models.sku_list import llama_meta_net_info, resolve_model
    from llama_models.cli.download import DownloadTask, ParallelDownloader

    meta_id = model.meta_model_id
    if not meta_id:
        raise ValueError(
            f"Model '{model.repo_id}' has source='meta' but no meta_model_id set. "
            "Check models.toml."
        )

    # Resolve the model and get its download manifest
    resolved = resolve_model(meta_id)
    if resolved is None:
        # Try with different casing conventions
        for variant in [meta_id, meta_id.title(), meta_id.replace("-", ".")]:
            resolved = resolve_model(variant)
            if resolved is not None:
                break
    if resolved is None:
        raise RuntimeError(
            f"Model '{meta_id}' not found in llama_models registry. "
            "Run `uv run llama-model list --show-all` to see valid IDs."
        )

    info = llama_meta_net_info(resolved)
    log.info(
        "Downloading %s from Meta: folder=%s, files=%s",
        model.repo_id,
        info.folder,
        info.files,
    )

    # Download into our structured directory
    local_dir = download_dir / model.org / model.name / "full"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Build download tasks — same URL construction as the official CLI:
    #   url = meta_url.replace("*", f"{info.folder}/{filename}")
    tasks = []
    for filename in info.files:
        output_file = str(local_dir / filename)
        if Path(output_file).exists():
            log.info("Already exists, skipping: %s", filename)
            continue
        url = signed_url.replace("*", f"{info.folder}/{filename}")
        total_size = info.pth_size if "consolidated" in filename else 0
        tasks.append(DownloadTask(
            url=url,
            output_file=output_file,
            total_size=total_size,
            max_retries=3,
        ))

    if tasks:
        downloader = ParallelDownloader(max_concurrent_downloads=3)
        asyncio.run(downloader.download_all(tasks))
    else:
        log.info("All files already downloaded.")

    file_count = sum(1 for f in local_dir.iterdir() if f.is_file())
    log.info("Meta download complete: %s (%d files in %s)", model.repo_id, file_count, local_dir)
    return local_dir


# ---------------------------------------------------------------------------
# HuggingFace Hub downloads
# ---------------------------------------------------------------------------


def _download_hf_full(model: ModelSpec, download_dir: Path) -> Path:
    """Download the full-weight checkpoint from HuggingFace Hub."""
    local_dir = download_dir / model.org / model.name / "full"
    local_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading full weights for %s from HuggingFace to %s", model.repo_id, local_dir)

    snapshot_download(
        repo_id=model.repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    log.info("Full-weight download complete: %s", local_dir)
    return local_dir


def _download_hf_gguf(model: ModelSpec, fmt: FormatSpec, download_dir: Path) -> Path:
    """Download GGUF quantized files from HuggingFace Hub."""
    local_dir = download_dir / model.org / model.name / "gguf"
    local_dir.mkdir(parents=True, exist_ok=True)

    source_repo = fmt.gguf_repo_id or model.repo_id

    for quant in fmt.quants:
        filename = f"{quant}.gguf"
        alt_filename = f"{model.name}-{quant}.gguf"
        alt_filename_2 = f"{model.name.lower()}-{quant.lower()}.gguf"

        downloaded = False
        for candidate in [filename, alt_filename, alt_filename_2]:
            try:
                log.info("Trying GGUF download: repo=%s file=%s", source_repo, candidate)
                path = hf_hub_download(
                    repo_id=source_repo,
                    filename=candidate,
                    local_dir=str(local_dir),
                )
                dest = local_dir / f"{quant}.gguf"
                downloaded_path = Path(path)
                if downloaded_path != dest:
                    downloaded_path.rename(dest)
                log.info("Downloaded GGUF %s -> %s", candidate, dest)
                downloaded = True
                break
            except Exception:
                log.debug("File %s not found in %s, trying next pattern", candidate, source_repo)
                continue

        if not downloaded:
            log.warning(
                "Could not download GGUF quant '%s' for %s from %s. "
                "You may need to set 'gguf_repo_id' in models.toml.",
                quant,
                model.repo_id,
                source_repo,
            )

    return local_dir


def _download_hf_gptq(model: ModelSpec, fmt: FormatSpec, download_dir: Path) -> Path:
    """Download GPTQ quantized weights from HuggingFace Hub."""
    local_dir = download_dir / model.org / model.name / "gptq" / fmt.variant
    local_dir.mkdir(parents=True, exist_ok=True)

    source_repo = fmt.gptq_repo_id or model.repo_id

    log.info(
        "Downloading GPTQ %s for %s from %s to %s",
        fmt.variant,
        model.repo_id,
        source_repo,
        local_dir,
    )

    snapshot_download(
        repo_id=source_repo,
        local_dir=str(local_dir),
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    log.info("GPTQ download complete: %s", local_dir)
    return local_dir
