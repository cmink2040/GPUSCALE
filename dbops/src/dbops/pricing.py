"""Automated GPU price collectors for vast.ai, RunPod, and eBay.

Produces `GpuPriceCreate` records ready to insert into `gpu_prices`. All
collectors normalize `gpu_name` to a short form (no "NVIDIA" / "GeForce"
prefix) so rows line up with what `benchmark_results.gpu_name` already
contains.

- Cloud rentals (vast, vast community, runpod) — per-hour prices via
  provider APIs, dumped for every GPU type the provider exposes.
- eBay (hardware one-time) — one-shot Browse API search per target GPU
  SKU, filtered to sellers with >=5000 positive feedback, cheapest
  remaining listing wins. Requires EBAY_CLIENT_ID and EBAY_CLIENT_SECRET
  env vars (app-only OAuth, client credentials grant).

Amazon prices stay manual (`dbops gpu-price add`) for now — adding
Amazon's Product Advertising API has an Associates-program gate and
different quotas that aren't worth the complexity yet.
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
# eBay (Browse API — client credentials grant)
# ---------------------------------------------------------------------------
#
# OAuth flow: POST client_id:client_secret (base64) to the token endpoint
# with grant_type=client_credentials and a scope, get an access token good
# for 2 hours, then use it as a Bearer on the Browse API. The free default
# quota is 5000 Browse calls per day — we'll use ~30 per collection run.
#
# Quality filters mirror the user's spec:
#   - seller.feedbackScore >= 5000
#   - seller.feedbackPercentage >= 98.0
#   - title doesn't contain obvious junk terms (parts, broken, box only, ...)
#   - condition = USED, buyingOptions includes FIXED_PRICE (no auctions),
#     price >= $20 (rules out listing-fee scam prices)

EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_SCOPES = "https://api.ebay.com/oauth/api_scope"

MIN_SELLER_FEEDBACK = 5000
MIN_SELLER_FEEDBACK_PCT = 98.0

# Substrings that disqualify a listing title (case-insensitive)
EBAY_TITLE_BLOCKLIST = {
    "box only", "empty box", "for parts", "as is", "as-is", "not working",
    "broken", "dead", "water damage", "water damaged", "burnt", "burned",
    "read description", "read desc", "no gpu", "repair only", "cracked",
    "bent", "mining", "as-parts", "won't boot", "wont boot", "fan only",
    "heatsink only", "shroud only", "pcb only",
}

# Target GPUs to search for on eBay. Each tuple is:
#   (gpu_name, vram_gb, search_query)
# gpu_name must match what we store elsewhere (normalized, no NVIDIA/GeForce).
# The search_query is what eBay's full-text engine sees — include VRAM so
# 16GB/32GB V100 variants stay separated.
EBAY_TARGETS: list[tuple[str, float, str]] = [
    # Blackwell consumer
    ("RTX 5090",    32, "Nvidia RTX 5090 32GB"),
    ("RTX 5080",    16, "Nvidia RTX 5080 16GB"),
    ("RTX 5070 Ti", 16, "Nvidia RTX 5070 Ti 16GB"),
    ("RTX 5070",    12, "Nvidia RTX 5070 12GB"),
    ("RTX 5060 Ti", 16, "Nvidia RTX 5060 Ti 16GB"),
    # Ada consumer
    ("RTX 4090",    24, "Nvidia RTX 4090 24GB"),
    ("RTX 4080",    16, "Nvidia RTX 4080 16GB"),
    ("RTX 4070 Ti", 12, "Nvidia RTX 4070 Ti 12GB"),
    ("RTX 4070",    12, "Nvidia RTX 4070 12GB"),
    ("RTX 4060 Ti", 16, "Nvidia RTX 4060 Ti 16GB"),
    # Ampere consumer
    ("RTX 3090 Ti", 24, "Nvidia RTX 3090 Ti 24GB"),
    ("RTX 3090",    24, "Nvidia RTX 3090 24GB"),
    ("RTX 3080 Ti", 12, "Nvidia RTX 3080 Ti 12GB"),
    ("RTX 3080",    10, "Nvidia RTX 3080 10GB"),
    ("RTX 3070 Ti",  8, "Nvidia RTX 3070 Ti 8GB"),
    ("RTX 3070",     8, "Nvidia RTX 3070 8GB"),
    ("RTX 3060 Ti",  8, "Nvidia RTX 3060 Ti 8GB"),
    ("RTX 3060",    12, "Nvidia RTX 3060 12GB"),
    # Turing consumer
    ("RTX 2080 Ti", 11, "Nvidia RTX 2080 Ti 11GB"),
    # Pascal / Volta / Turing datacenter
    ("Tesla P40",   24, "Nvidia Tesla P40 24GB"),
    ("Tesla P100",  16, "Nvidia Tesla P100 16GB"),
    ("Tesla V100",  16, "Nvidia Tesla V100 16GB PCIe"),
    ("Tesla V100",  32, "Nvidia Tesla V100 32GB SXM2"),
    ("Tesla T4",    16, "Nvidia Tesla T4 16GB"),
    # Ampere / Ada datacenter
    ("A100 PCIE",   40, "Nvidia A100 40GB PCIe"),
    ("A100 PCIE",   80, "Nvidia A100 80GB PCIe"),
    ("A40",         48, "Nvidia A40 48GB"),
    ("A10",         24, "Nvidia A10 24GB"),
    ("L4",          24, "Nvidia L4 24GB"),
    # Hopper datacenter — usually too expensive/rare on ebay but included
    ("H100 PCIE",   80, "Nvidia H100 80GB PCIe"),
]


def _ebay_mint_token() -> str:
    import base64
    import urllib.parse
    import urllib.request

    client_id = os.environ.get("EBAY_CLIENT_ID")
    client_secret = os.environ.get("EBAY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "EBAY_CLIENT_ID and EBAY_CLIENT_SECRET must be set in the "
            "environment. Register an app at developer.ebay.com and use the "
            "production keyset (not sandbox)."
        )
    cred = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode(
        {"grant_type": "client_credentials", "scope": EBAY_SCOPES}
    ).encode()
    req = urllib.request.Request(
        EBAY_OAUTH_URL,
        data=body,
        headers={
            "Authorization": f"Basic {cred}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "gpuscale-dbops/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"ebay token response missing access_token: {data}")
    return token


def _ebay_search(token: str, query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Hit item_summary/search for the given query and return raw item rows."""
    import urllib.parse
    import urllib.request

    params = {
        "q": query,
        # USED condition id is 3000. FIXED_PRICE excludes auctions.
        "filter": (
            "conditionIds:{3000},"
            "buyingOptions:{FIXED_PRICE},"
            "price:[20..],"
            "priceCurrency:USD"
        ),
        "sort": "price",
        "limit": str(limit),
    }
    url = EBAY_BROWSE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "gpuscale-dbops/1.0",
            # Narrow to US marketplace so we don't get listings in EUR/GBP
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data.get("itemSummaries") or []


def _valid_listing(item: dict[str, Any]) -> tuple[bool, str]:
    """Return (is_valid, reason) for an eBay item summary."""
    title = (item.get("title") or "").lower()
    for blocked in EBAY_TITLE_BLOCKLIST:
        if blocked in title:
            return False, f"title contains '{blocked}'"

    seller = item.get("seller") or {}
    fb_score = seller.get("feedbackScore")
    if fb_score is None or int(fb_score) < MIN_SELLER_FEEDBACK:
        return False, f"seller feedback {fb_score} < {MIN_SELLER_FEEDBACK}"

    fb_pct_raw = seller.get("feedbackPercentage") or "0"
    try:
        fb_pct = float(fb_pct_raw)
    except (TypeError, ValueError):
        fb_pct = 0.0
    if fb_pct < MIN_SELLER_FEEDBACK_PCT:
        return False, f"seller feedback {fb_pct}% < {MIN_SELLER_FEEDBACK_PCT}%"

    return True, "ok"


def collect_ebay_prices(
    targets: list[tuple[str, float, str]] | None = None,
) -> list[GpuPriceCreate]:
    """Search eBay for each target GPU, filter to reputable sellers,
    return the cheapest matching used listing per GPU.
    """
    targets_list = list(targets) if targets is not None else list(EBAY_TARGETS)
    token = _ebay_mint_token()

    out: list[GpuPriceCreate] = []
    for gpu_name, vram_gb, query in targets_list:
        try:
            items = _ebay_search(token, query, limit=50)
        except Exception:
            # One failing query shouldn't stop the batch.
            continue

        cheapest = None
        for item in items:
            ok, _reason = _valid_listing(item)
            if not ok:
                continue
            price = item.get("price") or {}
            try:
                value = float(price.get("value") or 0)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            if cheapest is None or value < cheapest["price"]:
                cheapest = {"price": value, "item": item}

        if cheapest is None:
            continue

        item = cheapest["item"]
        seller = item.get("seller") or {}
        out.append(
            GpuPriceCreate(
                gpu_name=gpu_name,
                gpu_vram_gb=float(vram_gb),
                source=PriceSource.EBAY,
                unit=PriceUnit.ONE_TIME,
                price_usd=cheapest["price"],
                listing_url=item.get("itemWebUrl"),
                seller=seller.get("username"),
                notes=(
                    f"Lowest eBay FIXED_PRICE used listing from a seller "
                    f">= {MIN_SELLER_FEEDBACK} feedback"
                ),
                raw_metadata={
                    "title": item.get("title"),
                    "condition": item.get("condition"),
                    "item_id": item.get("itemId"),
                    "seller_username": seller.get("username"),
                    "seller_feedback_score": seller.get("feedbackScore"),
                    "seller_feedback_percentage": seller.get("feedbackPercentage"),
                    "search_query": query,
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Aggregate collector
# ---------------------------------------------------------------------------


def collect_all_sources() -> dict[str, list[GpuPriceCreate]]:
    """Run every automated collector (vast DC + community, runpod, ebay) and
    return a map of source label -> rows.

    Failures in one collector don't block the others. If a collector raises,
    its per-source list is empty and the exception is stashed under
    `"<source>_error"`.
    """
    out: dict[str, list[GpuPriceCreate]] = defaultdict(list)
    for source_label, fn, kwargs in [
        ("vast", collect_vast_prices, {"community": False}),
        ("vast (community)", collect_vast_prices, {"community": True}),
        ("runpod", collect_runpod_prices, {}),
        ("ebay", collect_ebay_prices, {}),
    ]:
        try:
            out[source_label] = fn(**kwargs)
        except Exception as exc:
            out[source_label] = []
            out[f"{source_label}_error"] = [exc]  # type: ignore[assignment]
    return out


# Backward-compat alias — older callers may import the old name.
collect_all_cloud = collect_all_sources
