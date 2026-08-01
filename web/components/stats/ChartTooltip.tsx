"use client";

// Recharts' default tooltip ships hardcoded inline styles (white box, drop
// shadow) that fight this project's flat design language — this replaces
// it, styled like PlaceCard's existing card language.
interface TooltipPayloadEntry {
  name?: string;
  value?: number | string;
  color?: string;
}

export function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="bg-surface border border-line rounded-sm p-2 text-xs font-mono">
      {label && <div className="text-ink-faint mb-1">{label}</div>}
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-1.5">
          {entry.color && (
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{ background: entry.color }}
            />
          )}
          <span>{entry.name ?? ""}:</span>
          <span className="tabular-nums font-medium">{entry.value}</span>
        </div>
      ))}
    </div>
  );
}
