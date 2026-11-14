import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  allowedDevOrigins: ["*"],
  // The dev badge sat over the rail footer and the theme toggle.
  devIndicators: false,
};

export default nextConfig;
