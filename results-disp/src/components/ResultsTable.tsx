"use client";

import type { BenchmarkResult, SortConfig } from "@/lib/types";

interface ResultsTableProps {
  results: BenchmarkResult[];
  sort: SortConfig;
  onSort: (sort: SortConfig) => void;
  loading: boolean;
}

interface Column {
  key: keyof BenchmarkResult;
  label: string;
  format?: (value: unknown) => string;
  align?: "left" | "right";
}

const COLUMNS: Column[] = [
  { key: "gpu_name", label: "GPU" },
  { key: "gpu_vram_gb", label: "VRAM (GB)", align: "right" },
  { key: "gpu_count", label: "GPUs", align: "right" },
  { key: "provider", label: "Provider" },
  { key: "engine", label: "Engine" },
  { key: "model_name", label: "Model" },
  { key: "quantization", label: "Quant" },
  {
    key: "tokens_per_sec",
    label: "Tok/s",
    align: "right",
    format: (v) => (v != null ? Number(v).toFixed(1) : "-"),
  },
  {
    key: "time_to_first_token_ms",
    label: "TTFT (ms)",
    align: "right",
    format: (v) => (v != null ? Number(v).toFixed(0) : "-"),
  },
  {
    key: "prompt_eval_tokens_per_sec",
    label: "Prompt Tok/s",
    align: "right",
    format: (v) => (v != null ? Number(v).toFixed(1) : "-"),
  },
  {
    key: "peak_vram_mb",
    label: "Peak VRAM (MB)",
    align: "right",
    format: (v) => (v != null ? Number(v).toLocaleString() : "-"),
  },
  {
    key: "avg_power_draw_w",
    label: "Power (W)",
    align: "right",
    format: (v) => (v != null ? Number(v).toFixed(0) : "-"),
  },
  {
    key: "avg_gpu_util_pct",
    label: "Util %",
    align: "right",
    format: (v) => (v != null ? Number(v).toFixed(0) + "%" : "-"),
  },
  {
    key: "avg_gpu_temp_c",
    label: "Temp (C)",
    align: "right",
    format: (v) => (v != null ? Number(v).toFixed(0) : "-"),
  },
  {
    key: "total_wall_time_s",
    label: "Wall Time (s)",
    align: "right",
    format: (v) => (v != null ? Number(v).toFixed(1) : "-"),
  },
  { key: "host_os", label: "OS" },
];

function SortIndicator({
  column,
  sort,
}: {
  column: keyof BenchmarkResult;
  sort: SortConfig;
}) {
  if (sort.column !== column) {
    return <span className="ml-1 text-zinc-300 dark:text-zinc-600">&#8597;</span>;
  }
  return (
    <span className="ml-1 text-blue-500">
      {sort.direction === "asc" ? "\u2191" : "\u2193"}
    </span>
  );
}

export default function ResultsTable({
  results,
  sort,
  onSort,
  loading,
}: ResultsTableProps) {
  const handleSort = (column: keyof BenchmarkResult) => {
    if (sort.column === column) {
      onSort({
        column,
        direction: sort.direction === "asc" ? "desc" : "asc",
      });
    } else {
      onSort({ column, direction: "desc" });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-zinc-500 dark:text-zinc-400">
          Loading results...
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-zinc-500 dark:text-zinc-400">
          No results found. Try adjusting your filters.
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900">
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className={`cursor-pointer whitespace-nowrap px-3 py-2 font-medium text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 select-none transition-colors ${
                  col.align === "right" ? "text-right" : "text-left"
                }`}
              >
                {col.label}
                <SortIndicator column={col.key} sort={sort} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((row, idx) => (
            <tr
              key={row.id ?? idx}
              className="border-b border-zinc-100 hover:bg-zinc-50 dark:border-zinc-800/50 dark:hover:bg-zinc-900/50 transition-colors"
            >
              {COLUMNS.map((col) => {
                const raw = row[col.key];
                const display = col.format ? col.format(raw) : (raw ?? "-");
                return (
                  <td
                    key={col.key}
                    className={`whitespace-nowrap px-3 py-2 ${
                      col.align === "right" ? "text-right tabular-nums" : "text-left"
                    }`}
                  >
                    {String(display)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
