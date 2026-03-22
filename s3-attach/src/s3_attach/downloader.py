"""Download models from Meta's official distribution or HuggingFace Hub."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from s3_attach.config import FormatSpec, ModelSpec

log = logging.getLogger(__name__)

DEFAULT_DOWNLOAD_DIR = Path("downloads")

# Default location where `llama model download` stores checkpoints.
META_CHECKPOINTS_DIR = Path.home() / ".llama" / "checkpoints"


def download_model(
    model: ModelSpec,
    fmt: FormatSpec,
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
) -> Path:
    """Download a model in the specified format.

    Routes to Meta's official distribution or HuggingFace Hub based on
    the model's source config. GGUF and GPTQ formats always use HuggingFace
    since Meta only distributes full weights.

    Returns the local directory path containing the downloaded artifacts.
    """
    if fmt.type == "full" and model.source == "meta":
        return _download_meta_full(model, download_dir)
    elif fmt.type == "full":
        return _download_hf_full(model, download_dir)
    elif fmt.type == "gguf":
        return _download_hf_gguf(model, fmt, download_dir)
    elif fmt.type == "gptq":
        return _download_hf_gptq(model, fmt, download_dir)
    else:
        raise ValueError(f"Unknown format type: {fmt.type}")


# ---------------------------------------------------------------------------
# Meta official distribution (full weights only)
# ---------------------------------------------------------------------------


def _download_meta_full(model: ModelSpec, download_dir: Path) -> Path:
    """Download full weights via Meta's llama-models CLI.

    Uses `llama model download --source meta --model-id <id>`.
    The CLI downloads to ~/.llama/checkpoints/<model-id>/, then we copy
    the files into our structured download directory.

    Note: First-time use requires accepting Meta's license at
    https://llama.meta.com/ — you'll receive a signed URL via email
    that the CLI will prompt for.
    """
    meta_id = model.meta_model_id
    if not meta_id:
        raise ValueError(
            f"Model '{model.repo_id}' has source='meta' but no meta_model_id set. "
            "Check models.toml."
        )

    log.info("Downloading full weights for %s via Meta CLI (model-id: %s)", model.repo_id, meta_id)

    # Run the llama CLI
    cmd = ["llama", "model", "download", "--source", "meta", "--model-id", meta_id]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        log.debug("llama model download stdout: %s", result.stdout)
    except FileNotFoundError:
        raise RuntimeError(
            "The 'llama' CLI was not found. Install it with: pip install llama-models"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Meta download failed for {meta_id}.\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}\n"
            "If this is your first download, visit https://llama.meta.com/ to accept "
            "the license and obtain a signed URL."
        )

    # Meta downloads to ~/.llama/checkpoints/<model-id>/
    # The model-id in the directory may differ in casing from the CLI arg.
    # Search for a matching directory.
    meta_dir = _find_meta_checkpoint(meta_id)
    if not meta_dir or not meta_dir.exists():
        raise RuntimeError(
            f"Download appeared to succeed but checkpoint directory not found. "
            f"Expected under {META_CHECKPOINTS_DIR}/ for model-id '{meta_id}'. "
            f"Check `llama model list` for the correct model ID."
        )

    # Copy into our structured download directory
    local_dir = download_dir / model.org / model.name / "full"
    local_dir.mkdir(parents=True, exist_ok=True)

    log.info("Copying Meta checkpoint from %s to %s", meta_dir, local_dir)
    for src_file in meta_dir.iterdir():
        if src_file.is_file():
            dest = local_dir / src_file.name
            shutil.copy2(src_file, dest)
            log.debug("Copied %s", src_file.name)

    log.info("Meta full-weight download complete: %s", local_dir)
    return local_dir


def _find_meta_checkpoint(meta_model_id: str) -> Path | None:
    """Find the checkpoint directory for a given Meta model ID.

    The directory name under ~/.llama/checkpoints/ may use different casing
    or formatting than the CLI model-id, so we do a case-insensitive search.
    """
    if not META_CHECKPOINTS_DIR.exists():
        return None

    # Normalize for comparison: lowercase, strip hyphens/underscores/dots
    def normalize(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "").replace(".", "")

    target = normalize(meta_model_id)

    for entry in META_CHECKPOINTS_DIR.iterdir():
        if entry.is_dir() and normalize(entry.name) == target:
            return entry

    # Fallback: partial match
    for entry in META_CHECKPOINTS_DIR.iterdir():
        if entry.is_dir() and target in normalize(entry.name):
            return entry

    return None


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
    """Download GGUF quantized files from HuggingFace Hub.

    GGUF files are always sourced from HuggingFace (typically community quant repos)
    regardless of the model's primary source, since Meta does not distribute GGUF.
    """
    local_dir = download_dir / model.org / model.name / "gguf"
    local_dir.mkdir(parents=True, exist_ok=True)

    source_repo = fmt.gguf_repo_id or model.repo_id

    for quant in fmt.quants:
        # Convention: the GGUF file is named <model-name>-<quant>.gguf or just <quant>.gguf
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
                # Rename to the canonical name we use in the bucket (e.g. Q4_K_M.gguf)
                dest = local_dir / f"{quant}.gguf"
                downloaded_path = Path(path)
                if downloaded_path != dest:
                    downloaded_path.rename(dest)
                log.info("Downloaded GGUF %s -> %s", candidate, dest)
                downloaded = True
                break
            except Exception:
                log.debug(
                    "File %s not found in %s, trying next pattern", candidate, source_repo
                )
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
    """Download GPTQ quantized weights from HuggingFace Hub.

    GPTQ files are always sourced from HuggingFace regardless of the model's
    primary source, since Meta does not distribute GPTQ.
    """
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
