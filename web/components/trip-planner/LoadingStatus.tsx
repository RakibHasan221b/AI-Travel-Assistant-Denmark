"use client";

import { useEffect, useState } from "react";

// There's no SSE/streaming from the backend, so nothing about live
// per-agent progress can be real. Real (elapsed seconds) and approximate
// (which agent is "probably" running) are kept visually distinct, matching
// this project's honesty-over-confidence principle elsewhere.
function stageCopy(elapsedSeconds: number): string {
  if (elapsedSeconds < 20) return "Place Scout is searching the database for candidates…";
  if (elapsedSeconds < 50) return "Conditions Analyst is checking Copenhagen's weather…";
  return "Concierge is writing your recommendation…";
}

export function LoadingStatus({ onCancel }: { onCancel: () => void }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="mt-8 rounded-sm border border-line bg-surface p-6">
      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className="inline-block h-3 w-3 rounded-full bg-accent animate-pulse"
        />
        <p className="font-medium">{stageCopy(elapsed)}</p>
      </div>
      <p className="mt-1 text-xs text-ink-faint">
        Approximate — the agents run one after another; this just estimates where we are.
      </p>

      <div className="mt-4 flex items-center justify-between">
        <span className="font-mono text-sm text-ink-muted tabular-nums">
          {elapsed}s elapsed
        </span>
        <button
          type="button"
          onClick={onCancel}
          className="text-sm text-ink-muted underline hover:text-ink"
        >
          Cancel
        </button>
      </div>

      {elapsed > 90 && (
        <p className="mt-3 text-xs text-ink-faint">
          Still working — this is normal on the free-tier server, especially after a period of
          idle time.
        </p>
      )}
    </div>
  );
}
