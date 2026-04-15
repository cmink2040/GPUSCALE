// Cost + efficiency derivations for the /report page.
//
// Two assumptions drive everything here:
//
//  1. We use **nominal manufacturer TDP** for power math, not the noisy
//     avg_power_draw_w we record per benchmark run. Runtime measurements
//     on vast/runpod hosts are wildly inconsistent (SXM cards often
//     report 0-45W because nvidia-smi on HGX boards refuses to expose
//     per-GPU draw, datacenter hosts throttle during capture windows,
//     etc). For a "will this GPU blow up my electric bill?" answer the
//     nominal TDP is both deterministic and what buyers actually get
//     billed under load. The observed peak can still show up in a
//     tooltip if we want it later.
//
//  2. For a per-GPU row in the report we pick the **best** (highest)
//     tok/s we have observed across all engine / model / provider /
//     run combinations for that GPU. A GPU's top-of-leaderboard speed
//     is what makes the $/Mtok and energy/Mtok math meaningful as a
//     single number — averaging across runs would penalize GPUs that
//     also have slow runs on llama.cpp-CPU-fallback configurations.

import type { BenchmarkResult, GpuPrice } from "./types";

// Nominal manufacturer TDP in watts, keyed by normalized GPU identity.
// The key format is `<normalized name>|<round vram gb>` so two variants
// with the same family name (e.g. A100 PCIE 40GB vs 80GB) get distinct
// entries. Names are pre-normalized via the same prefix/suffix strip
// the price matcher uses.
const TDP_W: Record<string, number> = {
  // Consumer RTX 20/30/40/50
  "rtx 2080 ti|11": 250,
  "rtx 3060|12": 170,
  "rtx 3060 ti|8": 200,
  "rtx 3070|8": 220,
  "rtx 3080|10": 320,
  "rtx 3080 ti|12": 350,
  "rtx 3090|24": 350,
  "rtx 4060 ti|16": 160,
  "rtx 4070|12": 200,
  "rtx 4080|16": 320,
  "rtx 4090|24": 450,
  "rtx 5060 ti|16": 180,
  "rtx 5070|12": 250,
  "rtx 5080|16": 360,
  "rtx 5090|32": 575,
  // Workstation (RTX A-series / Quadro)
  "rtx a4000|16": 140,
  "rtx a5000|24": 230,
  "rtx a6000|48": 300,
  // Datacenter Ampere
  "a10|24": 150,
  "a40|48": 300,
  "a100 pcie|40": 250,
  "a100 pcie|80": 300,
  "a100 sxm4|80": 400,
  // Datacenter Ada / Hopper
  "l4|24": 72,
  "h100 sxm|80": 700,
  "h100 sxm5|80": 700,
  "h200 nvl|141": 600,
  // Datacenter Pascal / Volta / Turing
  "tesla p100|16": 250,
  "tesla p100|12": 250,
  "tesla v100|16": 250,
  "tesla v100|32": 300,
  "tesla t4|16": 70,
};

// Same normalizer as lib/prices.ts — duplicated here to avoid a circular
// dep and because the report page is the only other caller. If a third
// caller shows up, hoist this into a shared util.
export function normalizeGpuKey(name: string | null | undefined): string {
  if (!name) return "";
  let n = name.trim();
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
  n = n.replace(/[-\s](PCIE|SXM2|SXM4|SXM5|NVL|HBM2)[-\s]\d+GB$/i, "");
  return n.trim().toLowerCase();
}

export function vramKey(vram: number | null | undefined): number {
  if (vram == null) return 0;
  return Math.round(vram);
}

function tdpLookupKey(gpuName: string, vramGb: number): string {
  return `${normalizeGpuKey(gpuName)}|${vramKey(vramGb)}`;
}

export function lookupTdpW(gpuName: string, vramGb: number): number | null {
  return TDP_W[tdpLookupKey(gpuName, vramGb)] ?? null;
}

// ---------------------------------------------------------------------
// Derivations
// ---------------------------------------------------------------------

// Electricity cost per month, at the given rate and duty cycle.
//   tdpW:        watts (use nominal TDP)
//   hoursPerDay: 0-24, represents average runtime per day
//   ratePerKwh:  $/kWh (default 0.13)
// Returns dollars per 30-day month.
export function monthlyPowerCost(
  tdpW: number,
  hoursPerDay: number,
  ratePerKwh: number,
): number {
  const kwhPerMonth = (tdpW / 1000) * hoursPerDay * 30;
  return kwhPerMonth * ratePerKwh;
}

// Energy consumed to generate 1M tokens, in kWh.
//   tokPerSec must be > 0
export function kwhPer1MTokens(tdpW: number, tokPerSec: number): number {
  if (tokPerSec <= 0) return Infinity;
  const secondsPer1M = 1_000_000 / tokPerSec;
  const kwh = (tdpW * secondsPer1M) / (1000 * 3600);
  return kwh;
}

// Dollar cost of electricity per 1M tokens (ignores capex / rental).
export function electricityCostPer1MTokens(
  tdpW: number,
  tokPerSec: number,
  ratePerKwh: number,
): number {
  return kwhPer1MTokens(tdpW, tokPerSec) * ratePerKwh;
}

// Dollar cost per 1M tokens when renting.
//   dollarsPerHour × seconds / 3600 to get dollars-per-second,
//   then × 1_000_000 / tok/s to get dollars per 1M tokens.
// Simplifies to: ($/hr × 1_000_000) / (tok/s × 3600)
export function rentalCostPer1MTokens(
  dollarsPerHour: number,
  tokPerSec: number,
): number {
  if (tokPerSec <= 0) return Infinity;
  return (dollarsPerHour * 1_000_000) / (tokPerSec * 3600);
}

// Upfront cost per GB of VRAM. Useful for workloads where VRAM is the
// binding constraint (fitting larger contexts / models).
export function pricePerVramGb(upfrontUsd: number, vramGb: number): number {
  if (vramGb <= 0) return Infinity;
  return upfrontUsd / vramGb;
}

// Upfront cost per sustained tok/s. A "raw speed per dollar" number.
export function pricePerTokRate(upfrontUsd: number, tokPerSec: number): number {
  if (tokPerSec <= 0) return Infinity;
  return upfrontUsd / tokPerSec;
}

// Break-even runtime in hours between buying and renting.
//
// When you own the card you still pay electricity, so break-even is
// solved against the delta between rental $/hr and electricity $/hr:
//
//   hours = upfront / (rent $/hr - electricity $/hr)
//
// If electricity >= rental (rare but possible for huge idle-power
// cards vs a cheap spot rental) the break-even is never reached and
// we return Infinity.
export function breakEvenHours(
  upfrontUsd: number,
  rentalPerHour: number,
  tdpW: number,
  ratePerKwh: number,
): number {
  const electricityPerHour = (tdpW / 1000) * ratePerKwh;
  const delta = rentalPerHour - electricityPerHour;
  if (delta <= 0) return Infinity;
  return upfrontUsd / delta;
}

// ---------------------------------------------------------------------
// Row building
// ---------------------------------------------------------------------

// The shape each row of the report table renders from. Nulls propagate
// upward when a piece of data isn't available so the UI can show "—"
// rather than silently substituting a zero.
export interface CostReportRow {
  // Identity
  gpuKey: string;                        // normalized name|vram
  displayName: string;                   // best-guess user-facing name
  vramGb: number;
  // Inputs
  bestTokPerSec: number | null;          // highest tok/s observed for this GPU
  bestTokSource: string | null;          // "<engine> · <model> · <quant>" tag
  tdpW: number | null;
  upfrontUsd: number | null;
  upfrontSource: string | null;          // e.g. "ebay · egoods.supply · 2026-04-15"
  rentPerHour: number | null;
  rentSource: string | null;             // e.g. "vast (community) · 2026-04-14"
  // Derived (null whenever an input is missing)
  pricePerGb: number | null;
  pricePerTok: number | null;
  monthlyPowerUsd: number | null;
  kwhPer1MTok: number | null;
  electricityPer1MTok: number | null;
  rentalPer1MTok: number | null;
  breakEvenHrs: number | null;
}

// Cost-report config knobs controlled by the UI.
export interface CostConfig {
  ratePerKwh: number;    // $/kWh, default 0.13
  hoursPerDay: number;   // 0-24, default 24 (always-on)
}

export const DEFAULT_CONFIG: CostConfig = {
  ratePerKwh: 0.13,
  hoursPerDay: 24,
};

// Prettify the noisy driver-reported name into something the report
// can show as a heading. Strips NVIDIA/GeForce/Quadro but keeps "Tesla"
// and the form-factor suffix so "Tesla P100-PCIE-16GB" stays readable.
function prettifyName(name: string): string {
  let n = name.trim();
  for (const p of ["NVIDIA ", "GeForce ", "Quadro "]) {
    while (n.toLowerCase().startsWith(p.toLowerCase())) n = n.slice(p.length);
  }
  return n.trim();
}

const COST_SOURCES = new Set(["ebay", "amazon"]);
const RENTAL_SOURCES = new Set(["vast", "vast (community)", "runpod"]);

// Build one CostReportRow per (gpu key), folding together benchmark runs
// (best tok/s wins) and price rows (latest-per-source, then cheapest).
export function buildCostReport(
  benchmarks: BenchmarkResult[],
  prices: GpuPrice[],
  config: CostConfig,
): CostReportRow[] {
  // 1. Collapse benchmark rows to (gpu key) → best run.
  type Peak = {
    displayName: string;
    vramGb: number;
    tokPerSec: number;
    source: string;
  };
  const peaks = new Map<string, Peak>();
  for (const row of benchmarks) {
    // Skip CPU rows (they have sentinel VRAM values) — the cost report
    // is GPU-only by design.
    if (row.gpu_name?.toLowerCase().startsWith("cpu")) continue;
    if ((row.tokens_per_sec ?? 0) <= 0) continue;

    const key = `${normalizeGpuKey(row.gpu_name)}|${vramKey(row.gpu_vram_gb)}`;
    const tag = [row.engine, row.model_name, row.quantization]
      .filter(Boolean)
      .join(" · ");
    const existing = peaks.get(key);
    if (!existing || row.tokens_per_sec > existing.tokPerSec) {
      peaks.set(key, {
        displayName: prettifyName(row.gpu_name),
        vramGb: row.gpu_vram_gb,
        tokPerSec: row.tokens_per_sec,
        source: tag,
      });
    }
  }

  // 2. Collapse price rows to latest-per-source, then cheapest-per-bucket.
  //    The bucket key mirrors the peaks map so we can join below.
  const latestPrice = new Map<string, GpuPrice>();
  for (const p of prices) {
    const k = `${normalizeGpuKey(p.gpu_name)}|${vramKey(p.gpu_vram_gb)}|${p.source}`;
    const ex = latestPrice.get(k);
    if (!ex || new Date(p.collected_at) > new Date(ex.collected_at)) {
      latestPrice.set(k, p);
    }
  }
  const cheapest = new Map<
    string,
    { cost: GpuPrice | null; rental: GpuPrice | null }
  >();
  for (const p of latestPrice.values()) {
    const k = `${normalizeGpuKey(p.gpu_name)}|${vramKey(p.gpu_vram_gb)}`;
    const slot = cheapest.get(k) ?? { cost: null, rental: null };
    const price = Number(p.price_usd);
    if (COST_SOURCES.has(p.source)) {
      if (!slot.cost || price < Number(slot.cost.price_usd)) slot.cost = p;
    } else if (RENTAL_SOURCES.has(p.source)) {
      if (!slot.rental || price < Number(slot.rental.price_usd)) slot.rental = p;
    }
    cheapest.set(k, slot);
  }

  // 3. Union of keys so GPUs with pricing but no benchmarks, or vice
  //    versa, still show up in the report.
  const allKeys = new Set<string>([...peaks.keys(), ...cheapest.keys()]);

  // 4. Emit rows.
  const rows: CostReportRow[] = [];
  for (const key of allKeys) {
    const peak = peaks.get(key);
    const priceSlot = cheapest.get(key);
    // If there's no peak we can still emit a name from the price row.
    let displayName = peak?.displayName ?? "";
    let vramGb = peak?.vramGb ?? 0;
    if (!peak && priceSlot) {
      const ref = priceSlot.cost ?? priceSlot.rental;
      if (ref) {
        displayName = prettifyName(ref.gpu_name);
        vramGb = Number(ref.gpu_vram_gb);
      }
    }
    if (!displayName) continue; // nothing we can render

    const bestTokPerSec = peak?.tokPerSec ?? null;
    const bestTokSource = peak?.source ?? null;
    const tdpW = lookupTdpW(displayName, vramGb);
    const upfrontUsd = priceSlot?.cost ? Number(priceSlot.cost.price_usd) : null;
    const upfrontSource = priceSlot?.cost
      ? `${priceSlot.cost.source}${priceSlot.cost.seller ? ` · ${priceSlot.cost.seller}` : ""} · ${priceSlot.cost.collected_at.slice(0, 10)}`
      : null;
    const rentPerHour = priceSlot?.rental
      ? Number(priceSlot.rental.price_usd)
      : null;
    const rentSource = priceSlot?.rental
      ? `${priceSlot.rental.source} · ${priceSlot.rental.collected_at.slice(0, 10)}`
      : null;

    const pricePerGb =
      upfrontUsd != null && vramGb > 0 ? pricePerVramGb(upfrontUsd, vramGb) : null;
    const pricePerTok =
      upfrontUsd != null && bestTokPerSec != null
        ? pricePerTokRate(upfrontUsd, bestTokPerSec)
        : null;
    const monthlyPowerUsd =
      tdpW != null
        ? monthlyPowerCost(tdpW, config.hoursPerDay, config.ratePerKwh)
        : null;
    const kwhPer1MTok =
      tdpW != null && bestTokPerSec != null
        ? kwhPer1MTokens(tdpW, bestTokPerSec)
        : null;
    const electricityPer1MTok =
      tdpW != null && bestTokPerSec != null
        ? electricityCostPer1MTokens(tdpW, bestTokPerSec, config.ratePerKwh)
        : null;
    const rentalPer1MTok =
      rentPerHour != null && bestTokPerSec != null
        ? rentalCostPer1MTokens(rentPerHour, bestTokPerSec)
        : null;
    const breakEvenHrs =
      upfrontUsd != null && rentPerHour != null && tdpW != null
        ? breakEvenHours(upfrontUsd, rentPerHour, tdpW, config.ratePerKwh)
        : null;

    rows.push({
      gpuKey: key,
      displayName,
      vramGb,
      bestTokPerSec,
      bestTokSource,
      tdpW,
      upfrontUsd,
      upfrontSource,
      rentPerHour,
      rentSource,
      pricePerGb,
      pricePerTok,
      monthlyPowerUsd,
      kwhPer1MTok,
      electricityPer1MTok,
      rentalPer1MTok,
      breakEvenHrs,
    });
  }

  return rows;
}

// ---------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------

export function fmtUsd(n: number | null, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n >= 10000) return `$${(n / 1000).toFixed(1)}k`;
  if (n >= 1000) return `$${(n / 1000).toFixed(2)}k`;
  return `$${n.toFixed(digits)}`;
}

export function fmtUsdCents(n: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n < 0.01) return `$${n.toFixed(4)}`;
  if (n < 1) return `$${n.toFixed(3)}`;
  return `$${n.toFixed(2)}`;
}

export function fmtNumber(n: number | null, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

export function fmtHours(n: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n >= 24 * 365) return `${(n / (24 * 365)).toFixed(1)} yr`;
  if (n >= 24 * 30) return `${(n / (24 * 30)).toFixed(1)} mo`;
  if (n >= 24) return `${(n / 24).toFixed(1)} d`;
  return `${n.toFixed(0)} h`;
}
