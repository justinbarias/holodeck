import { defineCloudflareConfig } from "@opennextjs/cloudflare";

// No incremental cache: every route in this sample renders per request.
export default defineCloudflareConfig();
