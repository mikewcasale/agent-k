import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import { ThemeProvider } from "@/components/theme-provider";

import "./globals.css";
import { SessionProvider } from "next-auth/react";

export const metadata: Metadata = {
  metadataBase: new URL("https://agents-k.com"),
  title: "AGENT-K | Autonomous Multi-Agent Kaggle Competition System",
  description:
    "Discover, research, prototype, evolve, and submit winning solutions to Kaggle competitions using AI-powered multi-agent orchestration built with Pydantic-AI.",
  keywords: [
    "Kaggle",
    "AI agents",
    "machine learning",
    "data science",
    "competitions",
    "Pydantic-AI",
    "multi-agent",
    "automation",
  ],
  authors: [{ name: "Mike Casale", url: "https://github.com/mikewcasale" }],
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://agents-k.com",
    siteName: "AGENT-K",
    title: "AGENT-K | Autonomous Multi-Agent Kaggle Competition System",
    description:
      "Discover, research, prototype, evolve, and submit winning solutions to Kaggle competitions using AI-powered multi-agent orchestration.",
    images: [
      {
        url: "/images/og-image.png",
        width: 1200,
        height: 630,
        alt: "AGENT-K - Autonomous Multi-Agent Kaggle Competition System",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "AGENT-K | Autonomous Multi-Agent Kaggle Competition System",
    description:
      "AI-powered multi-agent system for winning Kaggle competitions.",
    images: ["/images/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport = {
  maximumScale: 1, // Disable auto-zoom on mobile Safari
};

const geist = Geist({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-geist",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-geist-mono",
});

const LIGHT_THEME_COLOR = "hsl(0 0% 100%)";
const DARK_THEME_COLOR = "hsl(240deg 10% 3.92%)";
const THEME_COLOR_SCRIPT = `\
(function() {
  var html = document.documentElement;
  var meta = document.querySelector('meta[name="theme-color"]');
  if (!meta) {
    meta = document.createElement('meta');
    meta.setAttribute('name', 'theme-color');
    document.head.appendChild(meta);
  }
  function updateThemeColor() {
    var isDark = html.classList.contains('dark');
    meta.setAttribute('content', isDark ? '${DARK_THEME_COLOR}' : '${LIGHT_THEME_COLOR}');
  }
  var observer = new MutationObserver(updateThemeColor);
  observer.observe(html, { attributes: true, attributeFilter: ['class'] });
  updateThemeColor();
})();`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      className={`${geist.variable} ${geistMono.variable}`}
      // `next-themes` injects an extra classname to the body element to avoid
      // visual flicker before hydration. Hence the `suppressHydrationWarning`
      // prop is necessary to avoid the React hydration mismatch warning.
      // https://github.com/pacocoursey/next-themes?tab=readme-ov-file#with-app
      lang="en"
      suppressHydrationWarning
    >
      <head>
        <script
          // biome-ignore lint/security/noDangerouslySetInnerHtml: "Required"
          dangerouslySetInnerHTML={{
            __html: THEME_COLOR_SCRIPT,
          }}
        />
      </head>
      <body className="antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          disableTransitionOnChange
          enableSystem
        >
          <Toaster position="top-center" />
          <SessionProvider>{children}</SessionProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
