// Attach matched gpu_prices rows to display rows.
//
// `gpu_prices` stores normalized GPU names (e.g. "RTX 3090", "Tesla V100"),
// while `benchmark_results.gpu_name` may carry vendor prefixes
// ("NVIDIA GeForce RTX 3090") depending on where the probe ran. We strip the
// same prefixes on both sides and then match on (name, round(vram_gb)).
//
// Per GPU we expose two prices to the UI:
//   - cost_price   : cheapest one-time listing across ebay / amazon
//   - rental_price : cheapest per-hour rate across vast / vast (community) / runpod
//
// The "latest" price per source is picked by collected_at DESC before the
// cross-source minimum, so a stale $100 V100 doesn't beat a current $250 one.

import type { DisplayRow, GpuPrice, PriceInfo } from "./types";

const COST_SOURCES = new Set(["ebay", "amazon"]);
const RENTAL_SOURCES = new Set(["vast", "vast (community)", "runpod"]);

function normalizeGpuKey(name: string | null | undefined): string {
  if (!name) return "";
  let n = name.trim();
  // Strip common vendor/brand prefixes that benchmark probes add but prices
  // don't. Apply repeatedly to catch "NVIDIA GeForce RTX 3090" → "RTX 3090".
  const prefixes = ["NVIDIA ", "GeForce ", "Quadro "];
  let changed = true;
  while (changed) {
    changed = false;
    for (const p of prefixes) {
      if (n.toLowerCase().startsWith(p.toLowerCase())) {
        n = n.slice(p.length);
        changed = true;
      }
    }
  }
  // Strip form-factor/VRAM suffixes that some drivers append:
  //   "Tesla P100-PCIE-16GB" → "Tesla P100"
  //   "Tesla V100 SXM2 32GB" → "Tesla V100"
  // The "Tesla " prefix is intentionally kept here because the gpu_prices rows
  // use it too (e.g. "Tesla P100", "Tesla V100"). Supports both dash and
  // space separators because drivers differ.
  n = n.replace(/[-\s](PCIE|SXM2|SXM4|SXM5|NVL|HBM2)[-\s]\d+GB$/i, "");
  return n.trim().toLowerCase();
}

function vramKey(vram: number | null | undefined): number {
  if (vram == null) return 0;
  return Math.round(vram);
}

// Build a map of (gpu_key, vram) -> latest-per-source prices, then collapse to
// the two PriceInfo slots we render.
export function buildPriceLookup(
  prices: GpuPrice[],
): Map<string, { cost: PriceInfo | null; rental: PriceInfo | null }> {
  // First pass: group by (gpu_key, vram, source) and keep only the most
  // recent row per bucket.
  const latest = new Map<string, GpuPrice>();
  for (const p of prices) {
    const key = `${normalizeGpuKey(p.gpu_name)}|${vramKey(p.gpu_vram_gb)}|${p.source}`;
    const existing = latest.get(key);
    if (!existing || new Date(p.collected_at) > new Date(existing.collected_at)) {
      latest.set(key, p);
    }
  }

  // Second pass: per (gpu_key, vram), pick the cheapest cost-source row and
  // the cheapest rental-source row.
  const out = new Map<string, { cost: PriceInfo | null; rental: PriceInfo | null }>();
  for (const row of latest.values()) {
    const gpuKey = `${normalizeGpuKey(row.gpu_name)}|${vramKey(row.gpu_vram_gb)}`;
    const slot = out.get(gpuKey) ?? { cost: null, rental: null };

    const info: PriceInfo = {
      price_usd: Number(row.price_usd),
      source: row.source,
      seller: row.seller,
      listing_url: row.listing_url,
      collected_at: row.collected_at,
    };

    if (COST_SOURCES.has(row.source)) {
      if (!slot.cost || info.price_usd < slot.cost.price_usd) slot.cost = info;
    } else if (RENTAL_SOURCES.has(row.source)) {
      if (!slot.rental || info.price_usd < slot.rental.price_usd) slot.rental = info;
    }

    out.set(gpuKey, slot);
  }

  return out;
}

export function attachPrices(
  rows: DisplayRow[],
  lookup: ReturnType<typeof buildPriceLookup>,
): DisplayRow[] {
  return rows.map((row) => {
    const key = `${normalizeGpuKey(row.gpu_name)}|${vramKey(row.gpu_vram_gb)}`;
    const match = lookup.get(key);
    return {
      ...row,
      cost_price: match?.cost ?? null,
      rental_price: match?.rental ?? null,
    };
  });
}

// Pretty-print a YYYY-MM-DD UTC date from an ISO timestamp.
export function formatCollectedAt(iso: string): string {
  try {
    const d = new Date(iso);
    const y = d.getUTCFullYear();
    const m = String(d.getUTCMonth() + 1).padStart(2, "0");
    const day = String(d.getUTCDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  } catch {
    return iso.slice(0, 10);
  }
}
