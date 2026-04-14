"use client";

import type { NormalizationOpts } from "@/lib/types";
import { NORMALIZATION_INFO } from "@/lib/normalize";

interface NormalizationsProps {
  opts: NormalizationOpts;
  onChange: (opts: NormalizationOpts) => void;
  rawCount: number;
  displayCount: number;
}

export default function Normalizations({
  opts,
  onChange,
  rawCount,
  displayCount,
}: NormalizationsProps) {
  const anyActive = opts.weight || opts.tier;
  const merged = anyActive ? rawCount - displayCount : 0;

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Normalize
          </span>
        </div>

        {(Object.keys(NORMALIZATION_INFO) as (keyof NormalizationOpts)[]).map((key) => {
          const info = NORMALIZATION_INFO[key];
          const active = opts[key];
          return (
            <label
              key={key}
              className={`group flex cursor-pointer items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors ${
                active
                  ? "bg-emerald-100 text-emerald-900 ring-1 ring-emerald-300 dark:bg-emerald-500/15 dark:text-emerald-200 dark:ring-emerald-400/30"
                  : "text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-900"
              }`}
              title={info.desc}
            >
              <input
                type="checkbox"
                checked={active}
                onChange={(e) => onChange({ ...opts, [key]: e.target.checked })}
                className="h-3.5 w-3.5 rounded border-zinc-400 text-emerald-500 focus:ring-emerald-500/50"
              />
              <span className="font-medium">{info.label}</span>
            </label>
          );
        })}

        {anyActive && (
          <div className="ml-auto text-xs text-zinc-500 dark:text-zinc-400 tabular-nums">
            {merged > 0 && (
              <span>
                merged <span className="font-semibold text-emerald-600 dark:text-emerald-400">{merged}</span> rows ({rawCount} → {displayCount})
              </span>
            )}
            {merged === 0 && <span>no merges</span>}
          </div>
        )}
      </div>
    </div>
  );
}
