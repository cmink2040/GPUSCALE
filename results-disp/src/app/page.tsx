"use client";

import { useCallback, useEffect, useState } from "react";
import Filters from "@/components/Filters";
import ResultsTable from "@/components/ResultsTable";
import { fetchResults, fetchDistinctValues } from "@/lib/queries";
import type { BenchmarkResult, Filters as FiltersType, SortConfig } from "@/lib/types";

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

export default function Home() {
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [filters, setFilters] = useState<FiltersType>(EMPTY_FILTERS);
  const [sort, setSort] = useState<SortConfig>(DEFAULT_SORT);
  const [filterOptions, setFilterOptions] = useState<FilterOptions>(EMPTY_OPTIONS);
  const [loading, setLoading] = useState(true);

  // Load filter options on mount
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

  return (
    <div className="flex flex-col min-h-screen">
      <header className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
        <div className="mx-auto max-w-[1600px] px-4 py-4 sm:px-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-zinc-900 dark:text-zinc-50">
                GPUSCALE Benchmark Leaderboard
              </h1>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                GPU inference benchmark results
              </p>
            </div>
            <div className="text-sm text-zinc-500 dark:text-zinc-400 tabular-nums">
              {!loading && (
                <span>
                  {results.length} result{results.length !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 mx-auto w-full max-w-[1600px] px-4 py-6 sm:px-6 flex flex-col gap-4">
        <Filters
          filters={filters}
          onFilterChange={setFilters}
          filterOptions={filterOptions}
        />
        <ResultsTable
          results={results}
          sort={sort}
          onSort={setSort}
          loading={loading}
        />
      </main>

      <footer className="border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
        <div className="mx-auto max-w-[1600px] px-4 py-3 sm:px-6">
          <p className="text-xs text-zinc-400 dark:text-zinc-500 text-center">
            GPUSCALE — Open GPU benchmark results
          </p>
        </div>
      </footer>
    </div>
  );
}
