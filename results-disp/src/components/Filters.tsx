"use client";

import type { Filters as FiltersType } from "@/lib/types";

interface FiltersProps {
  filters: FiltersType;
  onFilterChange: (filters: FiltersType) => void;
  filterOptions: {
    gpu_name: string[];
    model_name: string[];
    engine: string[];
    quantization: string[];
    provider: string[];
  };
}

const FILTER_FIELDS: { key: keyof FiltersType; label: string }[] = [
  { key: "gpu_name", label: "GPU" },
  { key: "model_name", label: "Model" },
  { key: "engine", label: "Engine" },
  { key: "quantization", label: "Quantization" },
  { key: "provider", label: "Provider" },
];

export default function Filters({
  filters,
  onFilterChange,
  filterOptions,
}: FiltersProps) {
  const handleChange = (key: keyof FiltersType, value: string) => {
    onFilterChange({ ...filters, [key]: value });
  };

  const handleReset = () => {
    onFilterChange({
      gpu_name: "",
      model_name: "",
      engine: "",
      quantization: "",
      provider: "",
    });
  };

  const hasActiveFilters = Object.values(filters).some(Boolean);

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <div className="flex flex-wrap items-end gap-3">
        {FILTER_FIELDS.map(({ key, label }) => (
          <div key={key} className="flex min-w-[9rem] flex-col gap-1.5">
            <label
              htmlFor={key}
              className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400"
            >
              {label}
            </label>
            <select
              id={key}
              value={filters[key]}
              onChange={(e) => handleChange(key, e.target.value)}
              className="h-9 rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 shadow-sm transition-colors hover:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:hover:border-zinc-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/60"
            >
              <option value="">All</option>
              {filterOptions[key].map((val) => (
                <option key={val} value={val}>
                  {val}
                </option>
              ))}
            </select>
          </div>
        ))}
        {hasActiveFilters && (
          <button
            onClick={handleReset}
            className="h-9 self-end rounded-md border border-zinc-300 bg-zinc-50 px-3 text-sm font-medium text-zinc-700 shadow-sm transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
