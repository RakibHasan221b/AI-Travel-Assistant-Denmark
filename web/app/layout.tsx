import type { Metadata } from "next";
import { Schibsted_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Schibsted Grotesk: designed for the Norwegian media group Schibsted —
// genuine Nordic design lineage, not the Inter/Space Grotesk default.
const display = Schibsted_Grotesk({
  variable: "--font-display",
  subsets: ["latin"],
});

// Used specifically for measured values (quality score, distances, minutes)
// — a deliberate signal that a number is real and computed, not invented.
const dataMono = JetBrains_Mono({
  variable: "--font-data",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Trip Planner — AI Denmark Explorer",
  description:
    "Three AI agents plan a real Copenhagen trip, grounded in real places, weather, and quality scores.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${dataMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
