# CLAUDE.md — HoloDeck CopilotKit Frontend

## What This Is

A Next.js 16 generic frontend for any HoloDeck agent. It uses **CopilotKit v2** (`@copilotkit/react-core/v2`) with the **AG-UI protocol** to communicate with the HoloDeck backend (`holodeck serve`).

## Tech Stack

- **Next.js** 16.4 with App Router
- **React** 19.3
- **CopilotKit** v1.87 (`@copilotkit/react-core`, `@copilotkit/react-ui`, `@copilotkit/runtime`)
- **AG-UI Client** 1.0 (file uploads go out as AG-UI 1.0 `image`/`audio`/`video`/`document` parts with a base64 `data` source; the filename travels in `metadata.filename`)
- **Tailwind CSS** v4 (CSS-first config via `@import "tailwindcss"` — no `tailwind.config.js`)
- **TypeScript** 5

## Project Structure

```
src/
├── app/
│   ├── layout.tsx          # Root layout: <CopilotKit> provider, runtime config, self-hosted fonts
│   ├── page.tsx            # Chat page: rail, header, conversation on the cue margin, composer
│   ├── memory-demo/page.tsx # Every tool card state from fixtures, collapsed and expanded
│   ├── globals.css         # Design tokens (light + html.dark) and all .pb-* styles
│   └── api/copilotkit/
│       └── route.ts        # CopilotKit API route (proxies to HoloDeck backend)
├── components/
│   ├── chat/               # Rail, Conversation (cue margin, brackets, groups), Composer, HowToRead
│   ├── tool-card/          # ToolCardShell (shared progressive disclosure), context, CopyButton
│   ├── memory/             # memory_timeline / memory_search / memory_provenance views + pure view logic
│   ├── ToolCallCard.tsx    # Generic tool card
│   ├── ToolCallRenderer.tsx # ToolCallView: picks the view for one tool call
│   ├── Icons.tsx           # Inline SVG icons (stroke 1.5)
│   ├── ThemeProvider.tsx   # next-themes wrapper
│   └── ThemeToggle.tsx     # Dark/light toggle
└── lib/
    ├── conversation.ts     # AG-UI messages -> rows: cue numbers, answer brackets, groups of >3 calls
    ├── use-run-events.ts   # Tool call start/end times and RUN_ERROR from agent.subscribe
    ├── tool-view.ts        # Headlines, failure detection, sizes, durations
    ├── supported-mime.ts   # File types holodeck serve accepts
    └── runtime-config.ts   # Runtime environment configuration (server + client)
```

The page draws the conversation itself from `agent.messages` (it does not use
`CopilotChat.View`), so it can number tool calls in the cue margin. Assistant
markdown still goes through `CopilotChatAssistantMessage.MarkdownRenderer`.
Styles are plain `.pb-*` classes with CSS custom properties; Tailwind
utilities are not used.

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
| `FILE_ACCEPT` | MIME types for file upload | Every type `holodeck serve` accepts (see `src/lib/supported-mime.ts`) |

### CopilotKit v2 API
- `useAgent` and `useCopilotKit` from `@copilotkit/react-core/v2`; messages are sent with `agent.addMessage` + `copilotkit.runAgent`
- CSS imported via `@copilotkit/react-core/v2/styles.css` in `page.tsx` (for the markdown renderer)

### Theming — IMPORTANT
CopilotKit (v1.54 and later) uses a **shadcn/ui-style theming system** with CSS variables. The old `--copilot-kit-*` variables (pre-v1.53) no longer work.

CopilotKit's CSS bundles its own **full Tailwind v4 build** which sets theme variables via `@layer theme`. This means:

1. **Theme variables must use `!important`** to override CopilotKit's `@layer theme` defaults.
2. **Dark mode** uses `html.dark` class selector (controlled by next-themes).
3. Use the `.pb-*` classes and tokens in `globals.css` instead of Tailwind utilities; the app tokens are not CopilotKit names (`--action`, not `--primary`).

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
