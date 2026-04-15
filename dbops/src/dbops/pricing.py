"""GPU price collectors and the eBay candidate fetcher.

Two flavors of tooling live here:

1. Automated cloud collectors (vast, vast community, runpod) — per-hour
   prices via provider APIs, dumped for every GPU type the provider
   exposes. These are safe to auto-pick because the provider APIs
   return authoritative per-GPU rates.

2. eBay Browse API candidate fetcher — does NOT auto-pick. eBay
   listings are adversarial: a $10 "RTX 3060" from a 5K+ feedback
   seller is usually a for-parts/"READ DESCRIPTION" trap that no
   static filter can reliably catch. Instead, this module just fetches
   candidate listings (title, price, seller, description, shipping,
   URL) and returns them as structured records for an external AI
   agent to review. The agent reads each description, flags scams,
   picks the right listing, and calls `dbops gpu-price add` to record
   its choice. Human/agent judgment stays in the loop.

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
# quota is 5000 Browse calls per day — each ebay-candidates run burns
# ~1 + N calls (1 search + N getItem lookups for descriptions).
#
# Only one server-side pre-filter is applied at fetch time:
#
#     seller.feedbackScore >= MIN_SELLER_FEEDBACK (5000)
#
# This is NOT a "pick this listing" threshold — it's a noise floor that
# removes brand-new accounts and one-off sellers so the agent has fewer
# candidates to review. Everything else — title SKU match, description
# parsing, "for parts" traps, "box only", "read description" gotchas,
# suspiciously low prices — is the AI reviewer's job, because reliably
# detecting scams on eBay requires reading the description and reasoning
# about intent, which static regex cannot do safely.

EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_ITEM_URL = "https://api.ebay.com/buy/browse/v1/item/"
EBAY_SCOPES = "https://api.ebay.com/oauth/api_scope"

MIN_SELLER_FEEDBACK = 5000

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


def _ebay_get_item(token: str, item_id: str) -> dict[str, Any]:
    """Fetch full item details (including description + shipping) via getItem."""
    import urllib.parse
    import urllib.request

    # item_id often comes back as "v1|123|0" — URL-encode it for the path
    url = EBAY_ITEM_URL + urllib.parse.quote(item_id, safe="")
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "gpuscale-dbops/1.0",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _strip_html(html: str, max_chars: int = 1500) -> str:
    """Quick-and-dirty HTML → plaintext for descriptions the agent will read.

    Not a full parser — just enough to get readable text out of a typical
    eBay listing description (most are table layouts + free text).
    """
    import html as html_mod
    import re

    if not html:
        return ""
    # Drop script/style blocks wholesale
    txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace common block-level tags with newlines so structure survives
    txt = re.sub(r"<(br|/p|/div|/li|/tr|/h[1-6])[^>]*>", "\n", txt, flags=re.IGNORECASE)
    # Drop all other tags
    txt = re.sub(r"<[^>]+>", "", txt)
    # Decode entities (&amp; &lt; etc.)
    txt = html_mod.unescape(txt)
    # Collapse whitespace
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt).strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars].rstrip() + "…"
    return txt


def _price_from_item(price_dict: dict[str, Any] | None) -> float:
    if not price_dict:
        return 0.0
    try:
        return float(price_dict.get("value") or 0)
    except (TypeError, ValueError):
        return 0.0


def _shipping_from_item(item: dict[str, Any]) -> float | None:
    opts = item.get("shippingOptions") or []
    if not opts:
        return None
    first = opts[0] or {}
    cost = first.get("shippingCost") or {}
    try:
        return float(cost.get("value") or 0)
    except (TypeError, ValueError):
        return None


def fetch_ebay_candidates(
    gpu_name: str,
    vram_gb: float | None,
    *,
    query: str | None = None,
    limit: int = 10,
    with_description: bool = True,
) -> dict[str, Any]:
    """Fetch eBay candidate listings for review by an external AI agent.

    This does NOT pick a winner — it returns the top `limit` results from
    the cheapest-first search, each enriched with the full description
    (via getItem) if `with_description=True`, for an agent to reason over.

    Server-side pre-filter: seller.feedbackScore >= MIN_SELLER_FEEDBACK.
    No other filters — title SKU match, junk-term detection, scam
    recognition, price sanity are all the agent's responsibility.

    Returns a dict with `gpu_name`, `query`, `fetched_at`, and
    `candidates` (list of dicts) — shape is designed for JSON output.
    """
    from datetime import datetime, timezone

    if query is None:
        # Default query: include vram if we know it to separate 16/32 GB variants
        query = f"Nvidia {gpu_name}"
        if vram_gb:
            query += f" {int(vram_gb)}GB"

    token = _ebay_mint_token()
    items = _ebay_search(token, query, limit=max(limit * 2, 20))

    # Pre-filter by seller feedback floor — noise reduction only.
    kept: list[dict[str, Any]] = []
    for item in items:
        seller = item.get("seller") or {}
        fb_score = seller.get("feedbackScore")
        if fb_score is None or int(fb_score) < MIN_SELLER_FEEDBACK:
            continue
        kept.append(item)

    # Sort cheapest first (eBay already sorts by price but we re-sort to be safe)
    kept.sort(key=lambda it: _price_from_item(it.get("price")))
    kept = kept[:limit]

    candidates: list[dict[str, Any]] = []
    for rank, item in enumerate(kept, start=1):
        seller = item.get("seller") or {}
        price = _price_from_item(item.get("price"))
        shipping = _shipping_from_item(item)
        total = price + (shipping or 0.0)

        description = ""
        if with_description:
            try:
                full = _ebay_get_item(token, item.get("itemId") or "")
                description = _strip_html(full.get("description") or "")
                # getItem also has shortDescription for some listings
                if not description:
                    description = _strip_html(full.get("shortDescription") or "")
            except Exception as exc:
                description = f"[failed to fetch description: {exc}]"

        candidates.append(
            {
                "rank": rank,
                "price_usd": round(price, 2),
                "shipping_usd": round(shipping, 2) if shipping is not None else None,
                "total_usd": round(total, 2),
                "condition": item.get("condition"),
                "title": item.get("title"),
                "seller": {
                    "username": seller.get("username"),
                    "feedback_score": seller.get("feedbackScore"),
                    "feedback_percentage": seller.get("feedbackPercentage"),
                },
                "listing_url": item.get("itemWebUrl"),
                "item_id": item.get("itemId"),
                "description": description,
            }
        )

    return {
        "gpu_name": gpu_name,
        "vram_gb": float(vram_gb) if vram_gb else None,
        "query": query,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "seller_feedback_floor": MIN_SELLER_FEEDBACK,
        "note": (
            "eBay candidate listings for review by an AI agent. This tool "
            "does NOT pick a winner — a reviewer must read each description "
            "and catch parts-only / 'read description' / SKU mismatch / "
            "suspiciously-low-price traps manually, then pick one and "
            "insert via `dbops gpu-price add`."
        ),
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Aggregate collector
# ---------------------------------------------------------------------------


def collect_all_sources() -> dict[str, list[GpuPriceCreate]]:
    """Run every automated cloud collector and return source -> rows.

    eBay is NOT included — it requires AI review (see fetch_ebay_candidates).
    Failures in one collector don't block the others; the exception is
    stashed under `"<source>_error"`.
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


# Backward-compat alias — older callers may import the old name.
collect_all_cloud = collect_all_sources
