export interface BenchmarkResult {
  id: string;
  gpu_name: string;
  gpu_vram_gb: number;
  gpu_count: number;
  provider: string;
  engine: string;
  model_name: string;
  quantization: string;
  tokens_per_sec: number;
  time_to_first_token_ms: number;
  prompt_eval_tokens_per_sec: number;
  peak_vram_mb: number;
  avg_power_draw_w: number | null;
  avg_gpu_util_pct: number | null;
  avg_gpu_temp_c: number | null;
  total_wall_time_s: number;
  host_os: string | null;
  created_at: string;
}

export interface Filters {
  gpu_name: string;
  model_name: string;
  engine: string;
  quantization: string;
  provider: string;
}

export type SortDirection = "asc" | "desc";

export interface SortConfig {
  column: keyof BenchmarkResult;
  direction: SortDirection;
}

// Display-time augmentations attached by the normalization layer.
export interface DisplayRow extends BenchmarkResult {
  merged_count?: number;        // how many raw rows this represents (1 = unmerged)
  normalized?: boolean;         // true if tok/s was transformed
  original_tps?: number;        // raw tok/s prior to normalization, for tooltip
  transform_notes?: string;     // human description of applied transforms
}

// Toggleable normalization knobs. Adding a new toggle = adding a key here +
// handling it in `applyNormalizations`.
export interface NormalizationOpts {
  weight: boolean;   // normalize across model/quant by bytes-per-token ratio
  tier: boolean;     // boost community runs to datacenter-equivalent
}
