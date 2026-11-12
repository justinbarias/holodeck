/**
 * Runtime configuration for the HoloDeck CopilotKit frontend.
 *
 * All values are driven by environment variables at runtime (no NEXT_PUBLIC_ prefix).
 * The server reads process.env via getServerRuntimeConfig() and passes the client
 * subset to <RuntimeConfigProvider>; client components consume it via useRuntimeConfig().
 */

const DEFAULT_AGENT_ID = "my_agent_replace_me";
const DEFAULT_AGENT_TITLE = "HoloDeck Assistant";
const DEFAULT_AGENT_DESCRIPTION = "AI-powered assistant";
const DEFAULT_FILE_ACCEPT = "image/*,application/pdf,.doc,.docx,.txt";
const DEFAULT_HOLODECK_BACKEND_URL = "http://127.0.0.1:8000/awp";

/** Fields safe to expose to the browser. */
export type ClientRuntimeConfig = {
  agentId: string;
  agentTitle: string;
  agentDescription: string;
  fileAccept: string;
};

/** Full server-side config — includes backend URL used only by the proxy route. */
export type ServerRuntimeConfig = ClientRuntimeConfig & {
  holodeckBackendUrl: string;
};

export function getServerRuntimeConfig(): ServerRuntimeConfig {
  return {
    agentId: process.env.AGENT_ID ?? DEFAULT_AGENT_ID,
    agentTitle: process.env.AGENT_TITLE ?? DEFAULT_AGENT_TITLE,
    agentDescription: process.env.AGENT_DESCRIPTION ?? DEFAULT_AGENT_DESCRIPTION,
    fileAccept: process.env.FILE_ACCEPT ?? DEFAULT_FILE_ACCEPT,
    holodeckBackendUrl:
      process.env.HOLODECK_BACKEND_URL ?? DEFAULT_HOLODECK_BACKEND_URL,
  };
}

export function toClientRuntimeConfig(
  config: ServerRuntimeConfig,
): ClientRuntimeConfig {
  const { agentId, agentTitle, agentDescription, fileAccept } = config;
  return { agentId, agentTitle, agentDescription, fileAccept };
}
