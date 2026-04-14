"""Cloud GPU price collectors for vast.ai and RunPod.

Produces `GpuPriceCreate` records ready to insert into `gpu_prices`. Both
collectors normalize `gpu_name` to a short form (no "NVIDIA" / "GeForce"
prefix) so rows line up with what `benchmark_results.gpu_name` already
contains.

Hardware prices (ebay, amazon) are NOT automated — they come in via
`dbops gpu-price add`, typically entered by a human or AI agent after
finding the lowest reasonable listing per the rules spelled out in the
CLI help text.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from typing import Any

from dbops.models import GpuPriceCreate, PriceSource, PriceUnit

# ---------------------------------------------------------------------------
# GPU name normalization
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(r"^(NVIDIA\s+|GeForce\s+)+", re.IGNORECASE)


def normalize_gpu_name(name: str) -> str:
    """Strip "NVIDIA " / "GeForce " prefixes so names from vast, runpod, and
    benchmark_results all collapse to the same key.

    Examples:
        "NVIDIA GeForce RTX 4090" -> "RTX 4090"
        "NVIDIA H100 SXM"         -> "H100 SXM"
        "Tesla V100"              -> "Tesla V100"
        "RTX 4090"                -> "RTX 4090"
    """
    if not name:
        return name
    out = _PREFIX_RE.sub("", name.strip())
    return " ".join(out.split())


# ---------------------------------------------------------------------------
# vast.ai
# ---------------------------------------------------------------------------


def _vastai_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    api_key = os.environ.get("VAST_API_KEY", "")
    env = os.environ.copy()
    if api_key:
        env["VAST_API_KEY"] = api_key
    return subprocess.run(
        ["vastai", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _search_vast_offers(datacenter: bool) -> list[dict[str, Any]]:
    """Query vast.ai for current single-GPU offers on the given tier."""
    query = (
        "num_gpus=1 disk_space>=50 dph<=10.0 "
        "inet_down>200 reliability>0.95"
    )
    if datacenter:
        query += " datacenter=True"
    r = _vastai_cli("search", "offers", query, "-o", "dph", "--raw")
    if r.returncode != 0:
        raise RuntimeError(f"vastai search offers failed: {r.stderr}")
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"vastai returned non-JSON: {exc}\n{r.stdout[:500]}")
    # Drop already-rented offers
    return [o for o in data if not o.get("rented")]


def collect_vast_prices(community: bool = False) -> list[GpuPriceCreate]:
    """Return one GpuPriceCreate per distinct (gpu_name, vram_gb) with the
    lowest current dph_total across all available offers on the given tier.
    """
    offers = _search_vast_offers(datacenter=not community)

    # Group by (normalized name, vram-GB-bucket) and track the cheapest offer
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for o in offers:
        raw_name = o.get("gpu_name") or ""
        name = normalize_gpu_name(raw_name)
        if not name:
            continue
        # vast's offer.gpu_ram is in MiB — same unit fix as vast.py providers
        vram_mib = int(o.get("gpu_ram") or 0)
        vram_gb = round(vram_mib / 1024)
        dph = o.get("dph_total")
        if dph is None or vram_gb <= 0:
            continue
        key = (name, vram_gb)
        cur = best.get(key)
        if cur is None or dph < cur["dph_total"]:
            best[key] = o

    source = PriceSource.VAST_COMMUNITY if community else PriceSource.VAST
    out: list[GpuPriceCreate] = []
    for (name, vram_gb), offer in best.items():
        out.append(
            GpuPriceCreate(
                gpu_name=name,
                gpu_vram_gb=float(vram_gb),
                source=source,
                unit=PriceUnit.PER_HOUR,
                price_usd=float(offer["dph_total"]),
                notes=f"min of {len(offers)} offers on vast",
                raw_metadata={
                    "offer_id": offer.get("id"),
                    "host_id": offer.get("host_id"),
                    "driver_version": offer.get("driver_version"),
                    "cuda_max_good": offer.get("cuda_max_good"),
                    "reliability": offer.get("reliability2"),
                    "inet_down": offer.get("inet_down"),
                    "num_offers_sampled": len(offers),
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# RunPod
# ---------------------------------------------------------------------------

RUNPOD_API_URL = "https://api.runpod.io/graphql"
RUNPOD_GPU_TYPES_QUERY = """
query GpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    lowestPrice(input: { gpuCount: 1 }) {
      uninterruptablePrice
    }
  }
}
"""


def _runpod_graphql(query: str) -> dict[str, Any]:
    import urllib.request

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY not set")
    payload = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        RUNPOD_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # RunPod's WAF rejects the default "Python-urllib/..." UA with 403.
            "User-Agent": "gpuscale-dbops/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    data = json.loads(body)
    if "errors" in data:
        raise RuntimeError(f"runpod graphql errors: {data['errors']}")
    return data.get("data", {})


def collect_runpod_prices() -> list[GpuPriceCreate]:
    """Pull current lowest on-demand price per GPU type from the RunPod API."""
    data = _runpod_graphql(RUNPOD_GPU_TYPES_QUERY)
    gpu_types = data.get("gpuTypes") or []

    out: list[GpuPriceCreate] = []
    for gt in gpu_types:
        display = gt.get("displayName") or ""
        name = normalize_gpu_name(display)
        if not name:
            continue
        vram_gb = gt.get("memoryInGb")
        if not vram_gb:
            continue
        lowest = gt.get("lowestPrice") or {}
        price = lowest.get("uninterruptablePrice")
        if price is None or price <= 0:
            continue
        out.append(
            GpuPriceCreate(
                gpu_name=name,
                gpu_vram_gb=float(vram_gb),
                source=PriceSource.RUNPOD,
                unit=PriceUnit.PER_HOUR,
                price_usd=float(price),
                notes="lowestPrice.uninterruptablePrice from RunPod API",
                raw_metadata={
                    "id": gt.get("id"),
                    "displayName": display,
                    "secureCloud": gt.get("secureCloud"),
                    "communityCloud": gt.get("communityCloud"),
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Aggregate collectors
# ---------------------------------------------------------------------------


def collect_all_cloud() -> dict[str, list[GpuPriceCreate]]:
    """Run every cloud collector and return a map of source -> rows.

    Failures in one collector don't prevent the others from running. The
    per-source list is empty if the collector raised, and the exception
    is stashed under the key `"<source>_error"`.
    """
    out: dict[str, list[GpuPriceCreate]] = defaultdict(list)
    for source_label, fn, kwargs in [
        ("vast", collect_vast_prices, {"community": False}),
        ("vast (community)", collect_vast_prices, {"community": True}),
        ("runpod", collect_runpod_prices, {}),
    ]:
        try:
            out[source_label] = fn(**kwargs)
        except Exception as exc:
            out[source_label] = []
            out[f"{source_label}_error"] = [exc]  # type: ignore[assignment]
    return out
