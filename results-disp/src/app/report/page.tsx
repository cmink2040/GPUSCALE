"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { fetchGpuPrices, fetchResults } from "@/lib/queries";
import {
  buildCostReport,
  DEFAULT_CONFIG,
  fmtHours,
  fmtNumber,
  fmtUsd,
  fmtUsdCents,
  type CostConfig,
  type CostReportRow,
} from "@/lib/costs";
import type { BenchmarkResult, GpuPrice } from "@/lib/types";

// Column sort keys cover the derived numeric fields. `null` → unsorted.
type SortKey =
  | "displayName"
  | "bestTokPerSec"
  | "tdpW"
  | "upfrontUsd"
  | "pricePerGb"
  | "pricePerTok"
  | "monthlyPowerUsd"
  | "kwhPer1MTok"
  | "electricityPer1MTok"
  | "rentPerHour"
  | "rentalPer1MTok"
  | "breakEvenHrs";

type SortDir = "asc" | "desc";

interface SortState {
  key: SortKey;
  dir: SortDir;
}

const DEFAULT_SORT: SortState = { key: "rentalPer1MTok", dir: "asc" };

// Headline card kinds. Each one picks the "best" row by a specific metric.
interface Highlight {
  label: string;
  sub: string;
  row: CostReportRow | null;
  valueFmt: string;
}

function rowWith<K extends keyof CostReportRow>(
  rows: CostReportRow[],
  field: K,
  mode: "min" | "max",
): CostReportRow | null {
  let best: CostReportRow | null = null;
  for (const r of rows) {
    const v = r[field];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (!best) {
      best = r;
      continue;
    }
    const cur = best[field] as number;
    if (mode === "min" ? v < cur : v > cur) best = r;
  }
  return best;
}

export default function ReportPage() {
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([]);
  const [prices, setPrices] = useState<GpuPrice[]>([]);
  const [config, setConfig] = useState<CostConfig>(DEFAULT_CONFIG);
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const [b, p] = await Promise.all([
        fetchResults(
          { gpu_name: "", model_name: "", engine: "", quantization: "", provider: "" },
          { column: "tokens_per_sec", direction: "desc" },
        ),
        fetchGpuPrices(),
      ]);
      setBenchmarks(b);
      setPrices(p);
      setLoading(false);
    }
    load();
  }, []);

  const rows = useMemo(
    () => buildCostReport(benchmarks, prices, config),
    [benchmarks, prices, config],
  );

  const sortedRows = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sort.key];
      const bv = b[sort.key];
      // Nulls always sink to the bottom regardless of direction.
      const aNull = av == null || (typeof av === "number" && !Number.isFinite(av));
      const bNull = bv == null || (typeof bv === "number" && !Number.isFinite(bv));
      if (aNull && bNull) return 0;
      if (aNull) return 1;
      if (bNull) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return sort.dir === "asc" ? av - bv : bv - av;
      }
      const as = String(av);
      const bs = String(bv);
      return sort.dir === "asc" ? as.localeCompare(bs) : bs.localeCompare(as);
    });
    return copy;
  }, [rows, sort]);

  const highlights: Highlight[] = useMemo(() => {
    const byRentMtok = rowWith(rows, "rentalPer1MTok", "min");
    const byEnergyMtok = rowWith(rows, "electricityPer1MTok", "min");
    const byDollarPerGb = rowWith(rows, "pricePerGb", "min");
    const byBreakEven = rowWith(rows, "breakEvenHrs", "min");
    return [
      {
        label: "Cheapest rental per 1M tokens",
        sub: "Lowest vast/runpod $ to generate a million tokens at this GPU's best run.",
        row: byRentMtok,
        valueFmt: byRentMtok ? fmtUsdCents(byRentMtok.rentalPer1MTok) : "—",
      },
      {
        label: "Most energy-efficient per 1M tokens",
        sub: "If you already own the card — cheapest electricity burn per million tokens.",
        row: byEnergyMtok,
        valueFmt: byEnergyMtok ? fmtUsdCents(byEnergyMtok.electricityPer1MTok) : "—",
      },
      {
        label: "Cheapest $ per GB VRAM (buy)",
        sub: "For VRAM-bound workloads — best dollars per usable gigabyte, upfront.",
        row: byDollarPerGb,
        valueFmt: byDollarPerGb ? fmtUsd(byDollarPerGb.pricePerGb) : "—",
      },
      {
        label: "Fastest break-even vs rental",
        sub: "Hours of rental equivalent to the full buy price (after subtracting your electricity).",
        row: byBreakEven,
        valueFmt: byBreakEven ? fmtHours(byBreakEven.breakEvenHrs) : "—",
      },
    ];
  }, [rows]);

  return (
    <div className="flex flex-col min-h-screen">
      <header className="border-b border-zinc-200/80 dark:border-zinc-800/80 bg-gradient-to-b from-white to-zinc-50 dark:from-zinc-950 dark:to-zinc-950/60 backdrop-blur">
        <div className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 sm:py-8">
          <div className="flex items-center justify-between gap-6">
            <Link href="/" className="flex items-center gap-4 hover:opacity-80 transition-opacity">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-zinc-900 text-white shadow-sm dark:bg-white dark:text-zinc-900">
                <svg
                  viewBox="0 0 32 32"
                  fill="currentColor"
                  fillRule="evenodd"
                  aria-hidden="true"
                  className="h-6 w-6"
                >
                  <path d="M2 9a2 2 0 0 1 2-2h20v18H4a2 2 0 0 1-2-2V9zm22-.5h4v17h-4v-17zM10 20a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm0-1.8a2.2 2.2 0 1 1 0-4.4 2.2 2.2 0 0 1 0 4.4zM18 20a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm0-1.8a2.2 2.2 0 1 1 0-4.4 2.2 2.2 0 0 1 0 4.4z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-3xl">
                  GPUSCALE
                </h1>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Cost & efficiency report
                </p>
              </div>
            </Link>
            <nav className="flex items-center gap-1 text-sm">
              <Link
                href="/"
                className="rounded-lg px-3 py-2 font-medium text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-100 transition-colors"
              >
                Leaderboard
              </Link>
              <span className="rounded-lg px-3 py-2 font-medium text-zinc-900 bg-zinc-100 dark:bg-zinc-900 dark:text-zinc-50">
                Cost report
              </span>
            </nav>
          </div>
        </div>
      </header>

      <main className="flex-1 mx-auto w-full max-w-[1600px] px-4 py-6 sm:px-6 flex flex-col gap-6">
        <Controls config={config} onChange={setConfig} />

        <Highlights highlights={highlights} />

        <CostTable
          rows={sortedRows}
          sort={sort}
          onSort={setSort}
          loading={loading}
        />

        <Methodology config={config} />
      </main>

      <footer className="border-t border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-950/60 backdrop-blur">
        <div className="mx-auto max-w-[1600px] px-4 py-4 sm:px-6">
          <p className="text-xs text-zinc-400 dark:text-zinc-500 text-center">
            GPUSCALE — Cost report
          </p>
        </div>
      </footer>
    </div>
  );
}

// ----------------------------------------------------------------------
// Sub-components
// ----------------------------------------------------------------------

function Controls({
  config,
  onChange,
}: {
  config: CostConfig;
  onChange: (c: CostConfig) => void;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <div className="flex flex-wrap items-center gap-6 px-5 py-4">
        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          Assumptions
        </span>

        <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300">
          <span>Electricity</span>
          <input
            type="number"
            step="0.01"
            min="0"
            value={config.ratePerKwh}
            onChange={(e) =>
              onChange({ ...config, ratePerKwh: Number(e.target.value) || 0 })
            }
            className="w-20 rounded-md border border-zinc-300 bg-white px-2 py-1 text-right tabular-nums text-zinc-900 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50"
          />
          <span className="text-zinc-500 dark:text-zinc-400">$/kWh</span>
        </label>

        <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300">
          <span>Duty cycle</span>
          <input
            type="range"
            min="1"
            max="24"
            step="1"
            value={config.hoursPerDay}
            onChange={(e) =>
              onChange({ ...config, hoursPerDay: Number(e.target.value) })
            }
            className="w-36 accent-emerald-500"
          />
          <span className="tabular-nums min-w-[3rem] text-right">
            {config.hoursPerDay}h/day
          </span>
        </label>

        <button
          onClick={() => onChange(DEFAULT_CONFIG)}
          className="ml-auto rounded-md border border-zinc-200 px-3 py-1 text-xs font-medium text-zinc-600 hover:bg-zinc-50 dark:border-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-900"
        >
          Reset
        </button>
      </div>
    </div>
  );
}

function Highlights({ highlights }: { highlights: Highlight[] }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {highlights.map((h) => (
        <div
          key={h.label}
          className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950"
        >
          <div className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            {h.label}
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
              {h.valueFmt}
            </span>
          </div>
          <div className="mt-1 text-sm text-zinc-700 dark:text-zinc-300">
            {h.row ? (
              <span className="font-medium">
                {h.row.displayName}
                <span className="ml-1 text-zinc-500 dark:text-zinc-400">
                  ({h.row.vramGb} GB)
                </span>
              </span>
            ) : (
              <span className="text-zinc-400">no data</span>
            )}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
            {h.sub}
          </p>
        </div>
      ))}
    </div>
  );
}

interface ColDef {
  key: SortKey;
  label: string;
  sub?: string;
  render: (r: CostReportRow) => string;
  title?: (r: CostReportRow) => string | undefined;
  align?: "left" | "right";
}

const COLS: ColDef[] = [
  {
    key: "displayName",
    label: "GPU",
    align: "left",
    render: (r) => `${r.displayName}`,
  },
  {
    key: "bestTokPerSec",
    label: "Best tok/s",
    sub: "across runs",
    render: (r) => fmtNumber(r.bestTokPerSec),
    title: (r) => r.bestTokSource ?? undefined,
  },
  {
    key: "tdpW",
    label: "TDP (W)",
    render: (r) => (r.tdpW != null ? String(r.tdpW) : "—"),
  },
  {
    key: "upfrontUsd",
    label: "Upfront",
    sub: "buy",
    render: (r) => fmtUsd(r.upfrontUsd),
    title: (r) => r.upfrontSource ?? undefined,
  },
  {
    key: "pricePerGb",
    label: "$ / GB VRAM",
    render: (r) => fmtUsd(r.pricePerGb),
  },
  {
    key: "pricePerTok",
    label: "$ / tok/s",
    sub: "per sustained t/s",
    render: (r) => fmtUsd(r.pricePerTok),
  },
  {
    key: "monthlyPowerUsd",
    label: "Power / mo",
    sub: "at current duty",
    render: (r) => fmtUsd(r.monthlyPowerUsd, 0),
  },
  {
    key: "kwhPer1MTok",
    label: "kWh / Mtok",
    render: (r) =>
      r.kwhPer1MTok == null || !Number.isFinite(r.kwhPer1MTok)
        ? "—"
        : r.kwhPer1MTok.toFixed(2),
  },
  {
    key: "electricityPer1MTok",
    label: "Elec $ / Mtok",
    sub: "own the card",
    render: (r) => fmtUsdCents(r.electricityPer1MTok),
  },
  {
    key: "rentPerHour",
    label: "Rent $/hr",
    sub: "cheapest",
    render: (r) => fmtUsdCents(r.rentPerHour),
    title: (r) => r.rentSource ?? undefined,
  },
  {
    key: "rentalPer1MTok",
    label: "Rent $ / Mtok",
    sub: "cloud",
    render: (r) => fmtUsdCents(r.rentalPer1MTok),
  },
  {
    key: "breakEvenHrs",
    label: "Break-even",
    sub: "buy vs rent",
    render: (r) => fmtHours(r.breakEvenHrs),
  },
];

function CostTable({
  rows,
  sort,
  onSort,
  loading,
}: {
  rows: CostReportRow[];
  sort: SortState;
  onSort: (s: SortState) => void;
  loading: boolean;
}) {
  const handleSort = (key: SortKey) => {
    if (sort.key === key) {
      onSort({ key, dir: sort.dir === "asc" ? "desc" : "asc" });
    } else {
      // Numeric columns default to ascending (cheaper = better); name defaults to A→Z.
      onSort({ key, dir: key === "displayName" ? "asc" : "asc" });
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-zinc-200 bg-white py-20 dark:border-zinc-800 dark:bg-zinc-950">
        <span className="text-zinc-500 dark:text-zinc-400">Computing cost report…</span>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-200 bg-zinc-50/80 dark:border-zinc-800 dark:bg-zinc-900/60">
            <th className="whitespace-nowrap px-3 py-3 text-right font-semibold text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              #
            </th>
            <th className="whitespace-nowrap px-3 py-3 text-right font-semibold text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              VRAM
            </th>
            {COLS.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className={`whitespace-nowrap px-3 py-3 font-semibold text-xs uppercase tracking-wider text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 cursor-pointer select-none transition-colors ${
                  col.align === "left" ? "text-left" : "text-right"
                }`}
              >
                <div className="flex flex-col">
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    <SortGlyph active={sort.key === col.key} dir={sort.dir} />
                  </span>
                  {col.sub && (
                    <span className="text-[9px] font-normal normal-case text-zinc-400 dark:text-zinc-600">
                      {col.sub}
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr
              key={row.gpuKey}
              className="border-b border-zinc-100 last:border-b-0 hover:bg-zinc-50 dark:border-zinc-800/60 dark:hover:bg-zinc-900/40 transition-colors"
            >
              <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums text-zinc-400 dark:text-zinc-500">
                {idx + 1}
              </td>
              <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums text-zinc-500 dark:text-zinc-400">
                {row.vramGb}
              </td>
              {COLS.map((col) => {
                const value = col.render(row);
                const title = col.title?.(row);
                const isName = col.key === "displayName";
                return (
                  <td
                    key={col.key}
                    className={`whitespace-nowrap px-3 py-2 ${
                      col.align === "left"
                        ? "text-left font-medium text-zinc-900 dark:text-zinc-50"
                        : "text-right tabular-nums text-zinc-700 dark:text-zinc-300"
                    } ${isName ? "" : ""}`}
                    title={title}
                  >
                    {value}
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

function SortGlyph({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active) return <span className="text-zinc-300 dark:text-zinc-600">&#8597;</span>;
  return <span className="text-blue-500">{dir === "asc" ? "↑" : "↓"}</span>;
}

function Methodology({ config }: { config: CostConfig }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
        How to read this
      </h2>
      <div className="mt-3 grid gap-3 text-xs leading-relaxed text-zinc-600 dark:text-zinc-400 sm:grid-cols-2">
        <div>
          <b className="text-zinc-800 dark:text-zinc-200">TDP, not measured power.</b>{" "}
          Power math uses manufacturer nominal TDP. Per-run{" "}
          <code>avg_power_draw_w</code> readings are noisy on vast/runpod hosts
          (SXM boards often report 0–45W regardless of load), so TDP gives a more
          honest &ldquo;will this blow up my electric bill&rdquo; answer.
        </div>
        <div>
          <b className="text-zinc-800 dark:text-zinc-200">Best tok/s per GPU.</b>{" "}
          Each row uses the single highest tok/s observed across all
          engine/model/quant combinations for that GPU. Hover the cell to see
          which run it came from.
        </div>
        <div>
          <b className="text-zinc-800 dark:text-zinc-200">Rent $/hr.</b> Cheapest
          of the latest vast, vast (community), and runpod prices. Hover to see
          which source provided it.
        </div>
        <div>
          <b className="text-zinc-800 dark:text-zinc-200">Break-even.</b>{" "}
          <code>upfront / (rent$/hr − electricity$/hr)</code>. If electricity
          alone costs more than the rental rate the row shows &ldquo;—&rdquo;.
        </div>
        <div>
          <b className="text-zinc-800 dark:text-zinc-200">
            Power / month
          </b>{" "}
          = TDP × {config.hoursPerDay} h/day × 30 × ${config.ratePerKwh.toFixed(2)}
          /kWh. Adjust the controls above to match your usage.
        </div>
        <div>
          <b className="text-zinc-800 dark:text-zinc-200">
            kWh / Mtok
          </b>{" "}
          = TDP × (1,000,000 / tok/s) / 3,600,000. A lower number means the GPU
          converts a joule into more generated tokens.
        </div>
      </div>
    </div>
  );
}
