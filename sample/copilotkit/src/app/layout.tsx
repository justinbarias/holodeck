import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { CopilotKit } from "@copilotkit/react-core";
import { ThemeProvider } from "@/components/ThemeProvider";
import {
  getServerRuntimeConfig,
  toClientRuntimeConfig,
} from "@/lib/runtime-config";
import { RuntimeConfigProvider } from "@/lib/runtime-config-provider";
import "./globals.css";

export const dynamic = "force-dynamic";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const config = getServerRuntimeConfig();
  return {
    title: config.agentTitle,
    description: config.agentDescription,
  };
}

const COPILOTKIT_ENDPOINT = "/api/copilotkit";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const clientConfig = toClientRuntimeConfig(getServerRuntimeConfig());

  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider>
          <RuntimeConfigProvider value={clientConfig}>
            <CopilotKit
              runtimeUrl={COPILOTKIT_ENDPOINT}
              agent={clientConfig.agentId}
            >
              {children}
            </CopilotKit>
          </RuntimeConfigProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
