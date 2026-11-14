import type { Metadata, Viewport } from "next";
import { Courier_Prime, Public_Sans } from "next/font/google";
import { CopilotKit } from "@copilotkit/react-core";
import { ThemeProvider } from "@/components/ThemeProvider";
import {
  getServerRuntimeConfig,
  toClientRuntimeConfig,
} from "@/lib/runtime-config";
import { RuntimeConfigProvider } from "@/lib/runtime-config-provider";
import "./globals.css";

export const dynamic = "force-dynamic";

// Self-hosted at build time by next/font: no runtime Google Fonts request.
const publicSans = Public_Sans({
  variable: "--font-public-sans",
  subsets: ["latin"],
  weight: ["400", "600"],
  display: "swap",
});

const courierPrime = Courier_Prime({
  variable: "--font-courier-prime",
  subsets: ["latin"],
  weight: ["400", "700"],
  display: "swap",
});

export async function generateMetadata(): Promise<Metadata> {
  const config = getServerRuntimeConfig();
  return {
    title: config.agentTitle,
    description: config.agentDescription,
  };
}

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#1d2347" },
    { media: "(prefers-color-scheme: dark)", color: "#12162e" },
  ],
};

const COPILOTKIT_ENDPOINT = "/api/copilotkit";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const clientConfig = toClientRuntimeConfig(getServerRuntimeConfig());

  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${publicSans.variable} ${courierPrime.variable}`}
    >
      <body>
        <ThemeProvider>
          <RuntimeConfigProvider value={clientConfig}>
            <CopilotKit
              runtimeUrl={COPILOTKIT_ENDPOINT}
              agent={clientConfig.agentId}
              enableInspector={false}
            >
              {children}
            </CopilotKit>
          </RuntimeConfigProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
