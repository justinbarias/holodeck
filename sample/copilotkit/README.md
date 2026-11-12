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

All configuration is via environment variables in `.env.local` (dev) or container environment (Docker):

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `AGENT_ID` | Yes | Must match `name` in agent.yaml | `my_agent_replace_me` |
| `AGENT_TITLE` | Yes | Header and browser tab title | `My Agent` |
| `AGENT_DESCRIPTION` | Yes | Subtitle below title | `What my agent does` |
| `HOLODECK_BACKEND_URL` | No | HoloDeck serve endpoint | `http://127.0.0.1:8000/awp` |
| `FILE_ACCEPT` | No | MIME types for uploads | `image/*,application/pdf,.doc,.docx,.txt` |

**Note:** These variables do NOT use the `NEXT_PUBLIC_` prefix. They are injected at runtime via server-side rendering, making them compatible with Docker containers.

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

## Features

- **Streaming responses** — Real-time token streaming via AG-UI protocol
- **Tool call visualization** — See tool names, arguments, and results inline
- **Memory timeline view** — `memory_timeline` results render as a fact timeline with an as-of date slider; open http://localhost:3000/memory-demo to see every state from fixtures, with no backend
- **Multimodal uploads** — Attach images, PDFs, and documents (also supports clipboard paste)
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
npm run lint    # Run ESLint
npm test        # Unit tests for the memory views (node --test, no extra dependencies)
```
