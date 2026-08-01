"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/explore", label: "Explore" },
  { href: "/trip-planner", label: "Trip Planner" },
  { href: "/stats", label: "Stats" },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-10 border-b border-line bg-background">
      <div className="mx-auto max-w-4xl px-6 py-3 flex items-center justify-between">
        <Link href="/" className="text-sm text-ink-muted hover:text-ink">
          AI Denmark Explorer
        </Link>
        <nav className="flex gap-5">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`text-sm pb-1 border-b-2 ${
                  active
                    ? "text-ink border-accent"
                    : "text-ink-muted border-transparent hover:text-ink"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
