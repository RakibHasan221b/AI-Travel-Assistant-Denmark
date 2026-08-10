import type { ModelEvaluation } from "@/lib/types";
import { ChartCard } from "./ChartCard";

// Not a chart — reuses ChartCard purely for its existing card shell
// (border/padding/title styling) so this doesn't need its own duplicate
// card markup. The metrics themselves are static, already-established
// offline evaluation results (see api/main.py's ModelEvaluation
// docstring) — this component only renders what the API returns.
export function ModelEvaluationSection({ data }: { data: ModelEvaluation }) {
  return (
    <ChartCard title="Model Evaluation">
      <div className="space-y-4">
        {data.metrics.map((m) => (
          <div key={m.label}>
            <div className="flex items-baseline gap-2">
              <span className="text-sm font-medium">{m.label}</span>
              <span className="font-mono text-lg font-semibold tabular-nums">
                {m.value.toFixed(3)}
              </span>
            </div>
            <p className="text-xs text-ink-faint mt-1">{m.description}</p>
          </div>
        ))}
      </div>
    </ChartCard>
  );
}
