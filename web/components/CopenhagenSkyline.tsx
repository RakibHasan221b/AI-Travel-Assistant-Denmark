// Flat, geometric Copenhagen silhouette — Nyhavn's gabled row houses,
// Rundetaarn, a sailboat on the harbor. Built from simple shapes (rects,
// polygons, arcs), not intricate freehand paths, to match the flat Nordic
// design direction rather than reading as decorative illustration.
export function CopenhagenSkyline({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 800 220"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="A flat illustration of the Copenhagen harbor skyline: Nyhavn's row houses, Rundetaarn, and a sailboat."
      className={className}
    >
      {/* waterline */}
      <line x1="0" y1="168" x2="800" y2="168" stroke="var(--line-strong)" strokeWidth="1.5" />

      {/* Rundetaarn */}
      <g>
        <rect x="600" y="60" width="46" height="108" fill="var(--ink)" opacity="0.9" />
        <path d="M600 60 A23 18 0 0 1 646 60 Z" fill="var(--ink)" opacity="0.9" />
        <rect x="617" y="30" width="12" height="30" fill="var(--ink)" opacity="0.9" />
      </g>

      {/* Nyhavn row houses */}
      {[
        { x: 60, w: 62, h: 92, color: "var(--accent)" },
        { x: 122, w: 54, h: 108, color: "var(--ink)" },
        { x: 176, w: 58, h: 84, color: "var(--secondary)" },
        { x: 234, w: 50, h: 100, color: "var(--ink)" },
        { x: 284, w: 60, h: 78, color: "var(--accent)" },
      ].map((h) => (
        <g key={h.x} opacity="0.88">
          <rect x={h.x} y={168 - h.h} width={h.w} height={h.h} fill={h.color} />
          <polygon
            points={`${h.x - 4},${168 - h.h} ${h.x + h.w / 2},${168 - h.h - 22} ${h.x + h.w + 4},${168 - h.h}`}
            fill={h.color}
          />
        </g>
      ))}

      {/* sailboat */}
      <g opacity="0.9">
        <path d="M420 168 L520 168 L500 178 L440 178 Z" fill="var(--ink)" />
        <line x1="465" y1="168" x2="465" y2="118" stroke="var(--ink)" strokeWidth="2" />
        <polygon points="467,120 467,166 500,166" fill="var(--secondary)" />
      </g>

      {/* Little Mermaid, simplified: a seated figure on a rock */}
      <g opacity="0.85">
        <ellipse cx="710" cy="164" rx="30" ry="10" fill="var(--line-strong)" />
        <circle cx="712" cy="130" r="7" fill="var(--accent)" />
        <path d="M705 137 Q702 155 712 160 Q722 152 719 137 Z" fill="var(--accent)" />
      </g>
    </svg>
  );
}
