"use client";

import { formatCollectedAt } from "@/lib/prices";
import type { BenchmarkResult, DisplayRow, PriceInfo, SortConfig } from "@/lib/types";

interface ResultsTableProps {
  results: DisplayRow[];
  sort: SortConfig;
  onSort: (sort: SortConfig) => void;
  loading: boolean;
  topTps?: number;
}

type ColumnKey = keyof BenchmarkResult | "cost_price" | "rental_price";

interface Column {
  key: ColumnKey;
  label: string;
  format?: (value: unknown) => string;
  align?: "left" | "right";
  sortable?: boolean; // synthetic price columns are not (yet) sortable
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
    key: "cost_price",
    label: "Cost (USD)",
    align: "right",
    sortable: false,
  },
  {
    key: "rental_price",
    label: "Rent $/hr",
    align: "right",
    sortable: false,
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

function formatCostUSD(n: number): string {
  if (n >= 1000) return `$${(n / 1000).toFixed(n >= 10000 ? 1 : 2)}k`;
  return `$${n.toFixed(0)}`;
}

function formatRentalUSD(n: number): string {
  return `$${n.toFixed(n < 1 ? 3 : 2)}`;
}

function priceTooltip(info: PriceInfo, unitLabel: string): string {
  const parts: string[] = [];
  parts.push(`${info.source}: $${info.price_usd.toFixed(2)}${unitLabel}`);
  if (info.seller) parts.push(`seller: ${info.seller}`);
  parts.push(`collected ${formatCollectedAt(info.collected_at)}`);
  return parts.join(" · ");
}

function SortIndicator({
  column,
  sort,
}: {
  column: ColumnKey;
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

const RANK_STYLES: Record<number, string> = {
  1: "bg-amber-100 text-amber-900 ring-1 ring-amber-300 dark:bg-amber-500/20 dark:text-amber-200 dark:ring-amber-500/40",
  2: "bg-zinc-200 text-zinc-800 ring-1 ring-zinc-300 dark:bg-zinc-500/20 dark:text-zinc-200 dark:ring-zinc-400/40",
  3: "bg-orange-100 text-orange-900 ring-1 ring-orange-300 dark:bg-orange-500/20 dark:text-orange-200 dark:ring-orange-500/40",
};

export default function ResultsTable({
  results,
  sort,
  onSort,
  loading,
  topTps = 0,
}: ResultsTableProps) {
  const handleSort = (column: ColumnKey, sortable: boolean) => {
    // Synthetic columns (cost/rental price) aren't backed by a DB field yet,
    // so we skip sorting them rather than passing an invalid column to Supabase.
    if (!sortable) return;
    const key = column as keyof BenchmarkResult;
    if (sort.column === key) {
      onSort({
        column: key,
        direction: sort.direction === "asc" ? "desc" : "asc",
      });
    } else {
      onSort({ column: key, direction: "desc" });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
        <div className="text-zinc-500 dark:text-zinc-400">
          Loading results…
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="flex items-center justify-center py-20 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
        <div className="text-zinc-500 dark:text-zinc-400">
          No results found. Try adjusting your filters.
        </div>
      </div>
    );
  }

  const sortedByTpsDesc = sort.column === "tokens_per_sec" && sort.direction === "desc";

  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-200 bg-zinc-50/80 dark:border-zinc-800 dark:bg-zinc-900/60">
            <th className="whitespace-nowrap px-3 py-3 text-right font-semibold text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              #
            </th>
            {COLUMNS.map((col) => {
              const sortable = col.sortable !== false;
              return (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key, sortable)}
                  className={`whitespace-nowrap px-3 py-3 font-semibold text-xs uppercase tracking-wider text-zinc-500 select-none transition-colors ${
                    sortable
                      ? "cursor-pointer hover:text-zinc-900 dark:hover:text-zinc-100"
                      : "cursor-default"
                  } dark:text-zinc-400 ${
                    col.align === "right" ? "text-right" : "text-left"
                  }`}
                >
                  {col.label}
                  {sortable && <SortIndicator column={col.key} sort={sort} />}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {results.map((row, idx) => {
            const rank = idx + 1;
            const rankStyle = sortedByTpsDesc ? RANK_STYLES[rank] : undefined;
            const tps = Number(row.tokens_per_sec) || 0;
            const tpsFraction =
              topTps > 0 ? Math.max(0.02, Math.min(1, tps / topTps)) : 0;

            return (
              <tr
                key={row.id ?? idx}
                className="border-b border-zinc-100 last:border-b-0 hover:bg-zinc-50 dark:border-zinc-800/60 dark:hover:bg-zinc-900/40 transition-colors"
              >
                <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums">
                  <span
                    className={`inline-flex min-w-[2rem] items-center justify-center rounded-md px-2 py-0.5 text-xs font-semibold ${
                      rankStyle ?? "text-zinc-400 dark:text-zinc-500"
                    }`}
                  >
                    {rank}
                  </span>
                </td>
                {COLUMNS.map((col) => {
                  // Synthetic price columns: render a link + tooltip and move on.
                  if (col.key === "cost_price" || col.key === "rental_price") {
                    const info =
                      col.key === "cost_price" ? row.cost_price : row.rental_price;
                    const unitLabel = col.key === "cost_price" ? "" : "/hr";
                    return (
                      <td
                        key={col.key}
                        className="whitespace-nowrap px-3 py-2 text-right tabular-nums text-zinc-700 dark:text-zinc-300"
                      >
                        {info ? (
                          <a
                            href={info.listing_url ?? undefined}
                            target={info.listing_url ? "_blank" : undefined}
                            rel="noreferrer"
                            title={priceTooltip(info, unitLabel)}
                            className={`inline-flex items-center gap-1 ${
                              info.listing_url
                                ? "text-blue-600 hover:underline dark:text-blue-400"
                                : ""
                            }`}
                          >
                            {col.key === "cost_price"
                              ? formatCostUSD(info.price_usd)
                              : formatRentalUSD(info.price_usd)}
                            <span className="text-[9px] uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
                              {info.source.replace("vast (community)", "vast-c")}
                            </span>
                          </a>
                        ) : (
                          <span className="text-zinc-300 dark:text-zinc-600">—</span>
                        )}
                      </td>
                    );
                  }

                  const raw = row[col.key];
                  const display = col.format ? col.format(raw) : (raw ?? "-");
                  const isTps = col.key === "tokens_per_sec";
                  const isGpu = col.key === "gpu_name";
                  return (
                    <td
                      key={col.key}
                      className={`whitespace-nowrap px-3 py-2 ${
                        col.align === "right" ? "text-right tabular-nums" : "text-left"
                      } ${isTps ? "relative font-semibold text-zinc-900 dark:text-zinc-50" : "text-zinc-700 dark:text-zinc-300"}`}
                    >
                      {isTps && topTps > 0 && (
                        <span
                          aria-hidden
                          className="absolute inset-y-1 right-2 rounded-sm bg-gradient-to-l from-emerald-400/40 to-emerald-400/0 dark:from-emerald-400/25 dark:to-emerald-400/0 pointer-events-none"
                          style={{ width: `${tpsFraction * 60}%` }}
                        />
                      )}
                      <span
                        className="relative inline-flex items-center gap-1.5"
                        title={
                          isTps && row.normalized && row.transform_notes
                            ? `Normalized: ${row.transform_notes}${row.original_tps != null ? ` (raw ${row.original_tps.toFixed(1)})` : ""}`
                            : undefined
                        }
                      >
                        {String(display)}
                        {isTps && row.normalized && (
                          <span className="text-emerald-500 dark:text-emerald-400 text-[10px]">★</span>
                        )}
                        {isGpu && (row.merged_count ?? 1) > 1 && (
                          <span
                            className="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-200"
                            title={`Merged from ${row.merged_count} runs`}
                          >
                            ×{row.merged_count}
                          </span>
                        )}
                      </span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
