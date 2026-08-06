// Circular progress ring for recommendation_confidence — arc length
// proportional to the percentage, colored by a fixed status band (never
// themed, see globals.css). Color is never the only signal: the
// percentage stays in plain ink text (status color doesn't leak into
// text, per this project's own dataviz convention), and the real
// recommendation_label sits right below as the actual, spelled-out claim
// — the ring is a visual accent for that claim, not a replacement for it.
const SIZE = 64;
const STROKE = 5;
const RADIUS = (SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function bandColor(confidence: number): string {
  if (confidence > 65) return "var(--color-good)";
  if (confidence >= 40) return "var(--color-warning)";
  return "var(--color-critical)";
}

export function RecommendationRing({
  confidence,
  label,
}: {
  confidence: number;
  label: string | null;
}) {
  const clamped = Math.max(0, Math.min(100, confidence));
  const offset = CIRCUMFERENCE * (1 - clamped / 100);
  const color = bandColor(clamped);

  return (
    <div className="flex flex-col items-center shrink-0">
      <div className="relative" style={{ width: SIZE, height: SIZE }}>
        <svg
          width={SIZE}
          height={SIZE}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="-rotate-90"
        >
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={RADIUS}
            fill="none"
            stroke="var(--color-line)"
            strokeWidth={STROKE}
          />
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={RADIUS}
            fill="none"
            stroke={color}
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.5s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-base font-semibold tabular-nums text-ink">
            {Math.round(clamped)}%
          </span>
        </div>
      </div>
      {label && (
        <div className="mt-1.5 flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 rounded-full shrink-0"
            style={{ background: color }}
            aria-hidden="true"
          />
          <span className="text-[10px] tracking-widest uppercase text-ink-faint">
            {label}
          </span>
        </div>
      )}
    </div>
  );
}
