# HoloDeck CopilotKit Frontend

A generic, reusable chat frontend for any HoloDeck agent. Built with Next.js 16, CopilotKit v2, and AG-UI protocol.

## Quick Start

```bash
# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local — set AGENT_ID to match your agent.yaml name

# Start dev server
npm run dev
```

Open http://localhost:3000

## Configuration

All configuration is via environment variables: `.env.local` (dev), the container environment (Docker), or Worker variables (Cloudflare Workers):

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `AGENT_ID` | Yes | Must match `name` in agent.yaml | `my_agent_replace_me` |
| `AGENT_TITLE` | Yes | Header and browser tab title | `My Agent` |
| `AGENT_DESCRIPTION` | Yes | Subtitle below title | `What my agent does` |
| `HOLODECK_BACKEND_URL` | No | HoloDeck serve endpoint | `http://127.0.0.1:8000/awp` |
| `FILE_ACCEPT` | No | MIME types for uploads | Every type `holodeck serve` accepts (see `src/lib/supported-mime.ts`) |

**Note:** These variables do NOT use the `NEXT_PUBLIC_` prefix. They are injected at runtime via server-side rendering, making them compatible with Docker containers and Workers.

## Deployment

The sample has two deployment paths. They share the same source and the same variables; neither needs code changes.

| Path | Build | Run | Configuration |
|------|-------|-----|---------------|
| Node (or Docker) | `npm run build` | `node .next/standalone/server.js` | Process environment |
| Cloudflare Workers | `npm run build:workers` | `npm run deploy:workers` | `vars` and secrets in `wrangler.jsonc` |

## Docker

Build and run as a Docker container:

```bash
# Build the image
docker build -t holodeck-copilotkit .

# Run with environment variables
docker run -p 3000:3000 \
  -e AGENT_ID=my_agent \  # use the actual `name` from your agent.yaml
  -e AGENT_TITLE="My Agent" \
  -e AGENT_DESCRIPTION="What my agent does" \
  -e HOLODECK_BACKEND_URL=http://holodeck-agent:8080/awp \
  holodeck-copilotkit
```

Or use docker-compose (see root `docker-compose.yml`).

## Cloudflare Workers

The Workers build uses the [OpenNext Cloudflare adapter](https://opennext.js.org/cloudflare). It runs the same `next build` and packages the result as a Worker with static assets. `wrangler.jsonc` and `open-next.config.ts` hold the configuration; the Node path does not read them.

```bash
# Local preview in the Workers runtime (workerd), http://localhost:8787
cp .dev.vars.example .dev.vars   # set AGENT_ID and HOLODECK_BACKEND_URL
npm run preview:workers

# Deploy (needs a Cloudflare account: `npx wrangler login`)
npm run deploy:workers
```

Before you deploy:

- Set `name` and the `vars` block in `wrangler.jsonc`. The Worker copies `vars` and secrets into `process.env`, so the same variables apply. `.dev.vars` overrides them for local preview only.
- `HOLODECK_BACKEND_URL` must be an address Cloudflare can reach. `127.0.0.1` works only in local preview.
- The build bundles any `.env`, `.env.production` and `.env.local` files as defaults. Do not build with secrets in them.
- The Worker is about 5.8 MB compressed. This is above the 3 MiB limit of the Workers Free plan, so the deployment needs the Workers Paid plan (10 MiB).
- `/api/copilotkit` streams server-sent events from the backend as they arrive; the Worker does not buffer them.

## Features

- **Streaming responses** — Real-time token streaming via AG-UI protocol
- **Tool call visualization** — See tool names, arguments, and results inline
- **Memory timeline view** — `memory_timeline` results render as a fact timeline with an as-of date slider; open http://localhost:3000/memory-demo to see every state from fixtures, with no backend
- **Multimodal uploads** — Attach images, PDFs, Office documents and text files (also supports clipboard paste). Files of a type `holodeck serve` does not accept are not attached; a notice names them.
- **Dark/light mode** — System-aware theme toggle
- **Generic** — Works with any HoloDeck agent via environment variables

## How It Works

```
Browser → /api/copilotkit (Next.js route) → HoloDeck backend (holodeck serve)
```

The Next.js API route acts as a proxy, forwarding CopilotKit requests to the HoloDeck backend via the AG-UI `HttpAgent`.

## Commands

```bash
npm run dev     # Development server (port 3000)
npm run build   # Production build (standalone output)
npm run start   # Start production server
npm run build:workers    # Cloudflare Workers build (.open-next/)
npm run preview:workers  # Build, then serve locally in workerd
npm run deploy:workers   # Build, then deploy with Wrangler
npm run lint    # Run ESLint
npm test        # Unit tests for the memory views (node --test, no extra dependencies)
```
