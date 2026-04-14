// Normalization layer for benchmark results.
//
// Each toggle is a dimension we ignore when grouping rows. Once enabled, rows
// that share the same remaining dimensions are merged into a single display
// row, and tok/s (plus other metrics) are first transformed appropriately.
//
// Currently supported toggles:
//   - weight  → normalize tok/s to a reference workload (Llama-3.1-8B Q4_K_M)
//               by scaling by `workload_bytes / REF_BYTES`. Merges rows that
//               differ only in (model, quant).
//   - tier    → boost community runs by ~14% to estimate datacenter-equivalent
//               throughput. Merges rows that differ only in provider tier.
//
// Both can be enabled simultaneously, in which case rows for the same
// (gpu_name, engine) merge into one row.

import type { BenchmarkResult, DisplayRow, NormalizationOpts } from "./types";

// Effective bytes touched per generated token (model weights × bits-per-weight
// / 8). Used as the proxy for memory-bandwidth-bound decode throughput.
const WORKLOAD_BYTES_GB: Record<string, number> = {
  // family-quant key
  "llama3-8b|Q4_K_M": 4.6,
  "llama3-8b|Q5_K_M": 5.4,
  "llama3-8b|Q8_0": 8.5,
  "llama3-8b|FP16": 16.0,
  "llama3-8b|FP8": 8.0,
  "qwen25-7b|AWQ-INT4": 3.9,
  "qwen35-9b|FP16": 18.0,
  "qwen35-27b|Q4_K_M": 15.5,
  "qwen35-27b|FP16": 54.0,
  "gemma4-31b|Q4_K_M": 17.8,
};

// Reference workload — the most common single GPU bench in the dataset.
const REF_BYTES_GB = 4.6;

// Tier multiplier — empirically observed community penalty across same-GPU
// pairs in the current dataset:
//   3090: 140.2 DC / 117.6 comm = 1.19
//   4090: 158.6 DC / 139.2 comm = 1.14
//   5090: 234.6 DC / 195.5 comm = 1.20
//   2080 Ti: 84.4 DC / 83.6 comm = 1.01 (outlier — likely same host pool)
// Median ≈ 1.17, mean (excluding 2080 Ti) ≈ 1.18. Using 1.15 as a
// conservative round number that fits comfortably inside the observed range.
const COMMUNITY_TO_DC_BOOST = 1.15;

function modelFamily(modelName: string | null | undefined): string {
  if (!modelName) return "unknown";
  const m = modelName.toLowerCase();
  if (m.includes("llama-3.1-8b") || m.includes("llama3-8b")) return "llama3-8b";
  if (m.includes("qwen2.5-7b") || m.includes("qwen-2.5-7b")) return "qwen25-7b";
  if (m.includes("qwen3.5-9b")) return "qwen35-9b";
  if (m.includes("qwen3.5-27b")) return "qwen35-27b";
  if (m.includes("gemma-4-31b") || m.includes("gemma4-31b")) return "gemma4-31b";
  return "unknown";
}

function workloadBytes(row: BenchmarkResult): number {
  const fam = modelFamily(row.model_name);
  const key = `${fam}|${row.quantization}`;
  return WORKLOAD_BYTES_GB[key] ?? REF_BYTES_GB;
}

function isCommunity(provider: string | null | undefined): boolean {
  return (provider ?? "").toLowerCase().includes("community");
}

// Build the merge key from the dimensions we are NOT ignoring.
function mergeKey(row: BenchmarkResult, opts: NormalizationOpts): string {
  const parts: string[] = [row.gpu_name ?? "?", row.engine ?? "?"];
  // If weight normalization is OFF, the (model, quant) is part of the key.
  if (!opts.weight) {
    parts.push(row.model_name ?? "?", row.quantization ?? "?");
  }
  // If tier normalization is OFF, the provider is part of the key.
  if (!opts.tier) {
    parts.push(row.provider ?? "?");
  } else {
    // When tier-normalizing, collapse "vast.ai" and "vast.ai (community)" into one.
    parts.push("any");
  }
  // gpu_count is always meaningful
  parts.push(String(row.gpu_count ?? 1));
  return parts.join("|");
}

function transformRow(row: BenchmarkResult, opts: NormalizationOpts): DisplayRow {
  const original_tps = row.tokens_per_sec;
  let tps = row.tokens_per_sec;
  let promptTps = row.prompt_eval_tokens_per_sec;
  const transforms: string[] = [];

  // Weight normalization: scale to reference workload.
  if (opts.weight && tps != null) {
    const wb = workloadBytes(row);
    const factor = wb / REF_BYTES_GB;
    if (Math.abs(factor - 1) > 0.01) {
      tps = tps * factor;
      if (promptTps != null) promptTps = promptTps * factor;
      transforms.push(`×${factor.toFixed(2)} (weight)`);
    }
  }

  // Tier normalization: boost community to DC-equivalent.
  if (opts.tier && tps != null && isCommunity(row.provider)) {
    tps = tps * COMMUNITY_TO_DC_BOOST;
    if (promptTps != null) promptTps = promptTps * COMMUNITY_TO_DC_BOOST;
    transforms.push(`×${COMMUNITY_TO_DC_BOOST} (tier)`);
  }

  return {
    ...row,
    tokens_per_sec: tps,
    prompt_eval_tokens_per_sec: promptTps,
    merged_count: 1,
    normalized: transforms.length > 0,
    original_tps: transforms.length > 0 ? original_tps : undefined,
    transform_notes: transforms.length > 0 ? transforms.join(", ") : undefined,
  };
}

// Average a list of numbers, ignoring nulls.
function avg(values: (number | null | undefined)[]): number | null {
  const nums = values.filter((v): v is number => v != null);
  if (nums.length === 0) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function maxOf(values: (number | null | undefined)[]): number | null {
  const nums = values.filter((v): v is number => v != null);
  if (nums.length === 0) return null;
  return Math.max(...nums);
}

function mergeRows(rows: DisplayRow[]): DisplayRow {
  if (rows.length === 1) return rows[0];
  // Use the first row as template, average the numeric metrics.
  const head = rows[0];
  const tps = avg(rows.map((r) => r.tokens_per_sec));
  return {
    ...head,
    // For the displayed identity, when tier is collapsed we mark provider as "merged".
    provider: rows.every((r) => r.provider === head.provider) ? head.provider : "merged",
    model_name: rows.every((r) => r.model_name === head.model_name) ? head.model_name : "merged",
    quantization: rows.every((r) => r.quantization === head.quantization) ? head.quantization : "merged",
    tokens_per_sec: tps ?? 0,
    time_to_first_token_ms: avg(rows.map((r) => r.time_to_first_token_ms)) ?? 0,
    prompt_eval_tokens_per_sec: avg(rows.map((r) => r.prompt_eval_tokens_per_sec)) ?? 0,
    peak_vram_mb: avg(rows.map((r) => r.peak_vram_mb)) ?? 0,
    avg_power_draw_w: avg(rows.map((r) => r.avg_power_draw_w)),
    avg_gpu_util_pct: avg(rows.map((r) => r.avg_gpu_util_pct)),
    avg_gpu_temp_c: maxOf(rows.map((r) => r.avg_gpu_temp_c)),
    total_wall_time_s: avg(rows.map((r) => r.total_wall_time_s)) ?? 0,
    merged_count: rows.length,
    normalized: rows.some((r) => r.normalized),
    original_tps: undefined,
  };
}

// Apply normalizations + merge equivalent rows.
export function applyNormalizations(
  rows: BenchmarkResult[],
  opts: NormalizationOpts,
): DisplayRow[] {
  // Fast path: no toggles → just wrap as DisplayRow without changes.
  if (!opts.weight && !opts.tier) {
    return rows.map((r) => ({ ...r, merged_count: 1 }));
  }

  const transformed = rows.map((r) => transformRow(r, opts));

  // Bucket by merge key.
  const buckets = new Map<string, DisplayRow[]>();
  for (const row of transformed) {
    const key = mergeKey(row, opts);
    const list = buckets.get(key) ?? [];
    list.push(row);
    buckets.set(key, list);
  }

  return Array.from(buckets.values()).map(mergeRows);
}

// Re-export for the UI to know the constants.
export const NORMALIZATION_INFO = {
  weight: {
    label: "Normalize weights",
    desc: `Scale tok/s to a reference workload (Llama 3.1 8B Q4_K_M, ~${REF_BYTES_GB} GB/token). Merges runs that differ only in model/quant.`,
  },
  tier: {
    label: "Normalize community → DC",
    desc: `Boost community runs by ${COMMUNITY_TO_DC_BOOST}× to estimate datacenter-equivalent throughput. Merges community + DC runs.`,
  },
} as const;
