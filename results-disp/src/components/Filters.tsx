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
    <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex flex-wrap items-end gap-4">
        {FILTER_FIELDS.map(({ key, label }) => (
          <div key={key} className="flex flex-col gap-1">
            <label
              htmlFor={key}
              className="text-xs font-medium text-zinc-500 dark:text-zinc-400"
            >
              {label}
            </label>
            <select
              id={key}
              value={filters[key]}
              onChange={(e) => handleChange(key, e.target.value)}
              className="h-9 rounded-md border border-zinc-300 bg-white px-3 text-sm text-zinc-900 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
            className="h-9 rounded-md border border-zinc-300 bg-zinc-100 px-3 text-sm font-medium text-zinc-700 hover:bg-zinc-200 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700 transition-colors"
          >
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
