# CLAUDE.md — HoloDeck CopilotKit Frontend

## What This Is

A Next.js 16 generic frontend for any HoloDeck agent. It uses **CopilotKit v2** (`@copilotkit/react-core/v2`) with the **AG-UI protocol** to communicate with the HoloDeck backend (`holodeck serve`).

## Tech Stack

- **Next.js** 16.1.1 with App Router
- **React** 19.2.3
- **CopilotKit** v1.54.0 (`@copilotkit/react-core`, `@copilotkit/react-ui`, `@copilotkit/runtime`)
- **AG-UI Client** ^0.0.47
- **Tailwind CSS** v4 (CSS-first config via `@import "tailwindcss"` — no `tailwind.config.js`)
- **TypeScript** 5

## Project Structure

```
src/
├── app/
│   ├── layout.tsx          # Root layout — wraps app in <CopilotKit> provider, injects runtime config
│   ├── page.tsx            # Main chat page — uses CopilotChat.View (v2 API)
│   ├── globals.css         # Theme variables + CopilotKit CSS overrides
│   └── api/copilotkit/
│       └── route.ts        # CopilotKit API route (proxies to HoloDeck backend)
├── components/
│   ├── ThemeProvider.tsx    # next-themes wrapper
│   ├── ThemeToggle.tsx      # Dark/light mode toggle button
│   └── ToolCallRenderer.tsx # Wildcard tool call renderer (useRenderToolCall)
└── lib/
    └── runtime-config.ts   # Runtime environment configuration (server + client)
```

## Key Architecture Decisions

### Runtime Configuration (Docker-compatible)

Environment variables use **no `NEXT_PUBLIC_` prefix** — they are read server-side via `process.env` in `layout.tsx` and passed into the client tree via a React context provider.

This enables true runtime configuration in Docker containers (values don't need to be known at build time).

- **Server components**: Call `getServerRuntimeConfig()` from `@/lib/runtime-config`
- **Client components**: Call `useRuntimeConfig()` from `@/lib/runtime-config-provider`
- **Layout**: Reads server config, strips the backend URL via `toClientRuntimeConfig()`, and wraps children in `<RuntimeConfigProvider>`. Uses `export const dynamic = "force-dynamic"` + `generateMetadata()` for dynamic metadata.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_ID` | Must match `name` field in agent.yaml | `my_agent_replace_me` |
| `AGENT_TITLE` | Displayed in header and browser tab | `HoloDeck Assistant` |
| `AGENT_DESCRIPTION` | Agent description subtitle | `AI-powered assistant` |
| `HOLODECK_BACKEND_URL` | HoloDeck serve endpoint | `http://127.0.0.1:8000/awp` |
| `FILE_ACCEPT` | MIME types for file upload | `image/*,application/pdf,.doc,.docx,.txt` |

### CopilotKit v2 API
- Uses `CopilotChat.View` (not the v1 `<CopilotChat>` component)
- Imports from `@copilotkit/react-core/v2`
- CSS imported via `@copilotkit/react-core/v2/styles.css` in `page.tsx`

### CopilotChat.View API (v1.54.0)
The v2 `CopilotChat.View` uses **flat props**, not `inputProps`:
```tsx
<CopilotChat.View
  messages={messages}
  isRunning={isRunning}
  inputValue={inputValue}
  onInputChange={setInputValue}
  onSubmitMessage={handleSubmitMessage}
  onStop={handleStop}
/>
```

### Theming — IMPORTANT
CopilotKit v1.54 uses a **shadcn/ui-style theming system** with CSS variables. The old `--copilot-kit-*` variables (pre-v1.53) no longer work.

CopilotKit's CSS bundles its own **full Tailwind v4 build** which sets theme variables via `@layer theme`. This means:

1. **Theme variables must use `!important`** to override CopilotKit's `@layer theme` defaults.
2. **Dark mode** uses `html.dark` class selector (controlled by next-themes).
3. Use custom CSS classes (`.holodeck-*`) instead of Tailwind utilities for elements outside CopilotKit.

### Docker Support
- `next.config.ts` has `output: "standalone"` for optimized Docker builds
- Multi-stage `Dockerfile` (node:22-alpine): deps → build → standalone runner
- `.dockerignore` excludes node_modules, .next, .env files

## Commands

```bash
npm run dev     # Start dev server (port 3000)
npm run build   # Production build (standalone)
npm run lint    # ESLint
```

## Backend Connection

The app connects to HoloDeck via the CopilotKit API route at `/api/copilotkit`. The backend must be running:

```bash
holodeck serve path/to/agent.yaml
```

The agent ID must match across:
1. `agent.yaml` → `name: my_agent`
2. `.env.local` → `AGENT_ID=my_agent` (the placeholder default `my_agent_replace_me` will not match any real agent)
