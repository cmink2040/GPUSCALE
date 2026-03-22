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
