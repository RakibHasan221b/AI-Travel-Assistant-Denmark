export function ChartCard({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border border-line rounded-sm bg-surface p-5">
      <h3 className="font-semibold text-sm mb-4">{title}</h3>
      {children}
      {caption && <p className="text-xs text-ink-faint mt-3">{caption}</p>}
    </div>
  );
}
