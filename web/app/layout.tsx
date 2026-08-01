import type { Metadata } from "next";
import { Schibsted_Grotesk, JetBrains_Mono } from "next/font/google";
import { NavBar } from "@/components/NavBar";
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
  title: {
    default: "AI Denmark Explorer",
    template: "%s — AI Denmark Explorer",
  },
  description:
    "Real Copenhagen places, grounded in real data — search, plan a trip, and see the stats behind it.",
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
      <body className="min-h-full flex flex-col">
        <NavBar />
        {children}
      </body>
    </html>
  );
}
