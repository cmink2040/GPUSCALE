"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import Filters from "@/components/Filters";
import Normalizations from "@/components/Normalizations";
import ResultsTable from "@/components/ResultsTable";
import { fetchResults, fetchDistinctValues, fetchGpuPrices } from "@/lib/queries";
import { applyNormalizations } from "@/lib/normalize";
import { attachPrices, buildPriceLookup } from "@/lib/prices";
import type { BenchmarkResult, Filters as FiltersType, GpuPrice, NormalizationOpts, SortConfig } from "@/lib/types";

const EMPTY_FILTERS: FiltersType = {
  gpu_name: "",
  model_name: "",
  engine: "",
  quantization: "",
  provider: "",
};

const DEFAULT_SORT: SortConfig = {
  column: "tokens_per_sec",
  direction: "desc",
};

type FilterOptions = Record<keyof FiltersType, string[]>;

const EMPTY_OPTIONS: FilterOptions = {
  gpu_name: [],
  model_name: [],
  engine: [],
  quantization: [],
  provider: [],
};

const EMPTY_NORM: NormalizationOpts = { weight: false, tier: false };

export default function Home() {
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [prices, setPrices] = useState<GpuPrice[]>([]);
  const [filters, setFilters] = useState<FiltersType>(EMPTY_FILTERS);
  const [normOpts, setNormOpts] = useState<NormalizationOpts>(EMPTY_NORM);
  const [sort, setSort] = useState<SortConfig>(DEFAULT_SORT);
  const [filterOptions, setFilterOptions] = useState<FilterOptions>(EMPTY_OPTIONS);
  const [loading, setLoading] = useState(true);

  // Load filter options + price table on mount (both are small + static-ish).
  useEffect(() => {
    async function loadOptions() {
      const keys = Object.keys(EMPTY_FILTERS) as (keyof FiltersType)[];
      const results = await Promise.all(keys.map((k) => fetchDistinctValues(k)));
      const opts = {} as FilterOptions;
      keys.forEach((key, i) => {
        opts[key] = results[i];
      });
      setFilterOptions(opts);
    }
    loadOptions();
    fetchGpuPrices().then(setPrices);
  }, []);

  // Fetch results when filters or sort change
  const loadResults = useCallback(async () => {
    setLoading(true);
    const data = await fetchResults(filters, sort);
    setResults(data);
    setLoading(false);
  }, [filters, sort]);

  useEffect(() => {
    loadResults();
  }, [loadResults]);

  // Price lookup is derived purely from the fetched price rows.
  const priceLookup = useMemo(() => buildPriceLookup(prices), [prices]);

  // Apply normalizations (and possibly merge equivalent rows) before display,
  // then fold in matched prices so the table has a cell per row.
  const displayRows = useMemo(
    () => attachPrices(applyNormalizations(results, normOpts), priceLookup),
    [results, normOpts, priceLookup],
  );

  const topTps = displayRows.length > 0
    ? Math.max(...displayRows.map((r) => r.tokens_per_sec ?? 0))
    : 0;

  return (
    <div className="flex flex-col min-h-screen">
      <header className="border-b border-zinc-200/80 dark:border-zinc-800/80 bg-gradient-to-b from-white to-zinc-50 dark:from-zinc-950 dark:to-zinc-950/60 backdrop-blur">
        <div className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 sm:py-8">
          <div className="flex items-center justify-between gap-6">
            <div className="flex items-center gap-4">
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
                  Open GPU inference benchmark leaderboard
                </p>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <nav className="flex items-center gap-1 text-sm">
                <span className="rounded-lg px-3 py-2 font-medium text-zinc-900 bg-zinc-100 dark:bg-zinc-900 dark:text-zinc-50">
                  Leaderboard
                </span>
                <Link
                  href="/report"
                  className="rounded-lg px-3 py-2 font-medium text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-100 transition-colors"
                >
                  Cost report
                </Link>
              </nav>
              <div className="hidden sm:flex flex-col items-end gap-1">
                <div className="text-2xl font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
                  {loading ? "…" : displayRows.length}
                </div>
                <div className="text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  {displayRows.length === 1 ? "result" : "results"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 mx-auto w-full max-w-[1600px] px-4 py-6 sm:px-6 flex flex-col gap-4">
        <Normalizations
          opts={normOpts}
          onChange={setNormOpts}
          rawCount={results.length}
          displayCount={displayRows.length}
        />
        <Filters
          filters={filters}
          onFilterChange={setFilters}
          filterOptions={filterOptions}
        />
        <ResultsTable
          results={displayRows}
          sort={sort}
          onSort={setSort}
          loading={loading}
          topTps={topTps}
        />
      </main>

      <footer className="border-t border-zinc-200 dark:border-zinc-800 bg-white/60 dark:bg-zinc-950/60 backdrop-blur">
        <div className="mx-auto max-w-[1600px] px-4 py-4 sm:px-6">
          <p className="text-xs text-zinc-400 dark:text-zinc-500 text-center">
            GPUSCALE — Open GPU benchmark results
          </p>
        </div>
      </footer>
    </div>
  );
}
