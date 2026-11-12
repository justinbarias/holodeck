import { HttpAgent } from "@ag-ui/client";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";
import { getServerRuntimeConfig } from "@/lib/runtime-config";

const COPILOTKIT_ENDPOINT = "/api/copilotkit";

const serviceAdapter = new ExperimentalEmptyAdapter();

export const POST = async (req: NextRequest) => {
  const { agentId, holodeckBackendUrl } = getServerRuntimeConfig();

  const runtime = new CopilotRuntime({
    agents: {
      [agentId]: new HttpAgent({ url: holodeckBackendUrl }),
    },
  });

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: COPILOTKIT_ENDPOINT,
  });

  return handleRequest(req);
};
