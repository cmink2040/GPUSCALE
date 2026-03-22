import { getSupabase } from "./supabase";
import type { BenchmarkResult, Filters, SortConfig } from "./types";

const TABLE = "benchmark_results";

export async function fetchResults(
  filters: Filters,
  sort: SortConfig
): Promise<BenchmarkResult[]> {
  const supabase = getSupabase();
  let query = supabase.from(TABLE).select("*");

  if (filters.gpu_name) {
    query = query.eq("gpu_name", filters.gpu_name);
  }
  if (filters.model_name) {
    query = query.eq("model_name", filters.model_name);
  }
  if (filters.engine) {
    query = query.eq("engine", filters.engine);
  }
  if (filters.quantization) {
    query = query.eq("quantization", filters.quantization);
  }
  if (filters.provider) {
    query = query.eq("provider", filters.provider);
  }

  query = query.order(sort.column, { ascending: sort.direction === "asc" });
  query = query.limit(500);

  const { data, error } = await query;

  if (error) {
    console.error("Error fetching results:", error);
    return [];
  }

  return (data as BenchmarkResult[]) ?? [];
}

export async function fetchDistinctValues(
  column: keyof BenchmarkResult
): Promise<string[]> {
  const supabase = getSupabase();
  const { data, error } = await supabase
    .from(TABLE)
    .select(column)
    .order(column, { ascending: true });

  if (error) {
    console.error(`Error fetching distinct ${column}:`, error);
    return [];
  }

  const unique = [
    ...new Set(
      (data ?? []).map((row) => String((row as Record<string, unknown>)[column]))
    ),
  ];
  return unique.filter(Boolean);
}
