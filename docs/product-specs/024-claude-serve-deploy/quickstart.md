# Quickstart: Claude Backend Serve & Deploy

**Feature**: 024-claude-serve-deploy
**Date**: 2026-03-20

## Scenario 1: Serve a Claude Agent Locally

**Prerequisites**: Node.js >= 18, `ANTHROPIC_API_KEY` set

```yaml
# agent.yaml
name: claude-assistant
model:
  provider: anthropic
  name: claude-sonnet-4-20250514
instructions:
  inline: "You are a helpful assistant."
claude:
  permission_mode: acceptAll
  max_concurrent_sessions: 5
```

```bash
holodeck serve agent.yaml --port 8000 --protocol rest
```

**Expected**: Server starts, pre-flight validation passes, health endpoint returns:
```json
{"status": "healthy", "backend_ready": true, "backend_diagnostics": []}
```

**Test request**:
```bash
curl -X POST http://localhost:8000/agent/claude-assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

## Scenario 2: Build and Deploy a Claude Agent Container

```bash
# Dry-run to inspect generated Dockerfile
holodeck deploy build agent.yaml --dry-run
# Expected: Dockerfile shows Node.js installation, non-root user

# Build the container image
holodeck deploy build agent.yaml --tag latest

# Run locally
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  holodeck/claude-assistant:latest
```

**Expected**: Container starts, entrypoint validates Node.js + credentials, serves requests.

## Scenario 3: Secure Container Deployment (Proxy Pattern)

```bash
# Run with security hardening + proxy credential injection
docker run \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  --user 1000:1000 \
  -e ANTHROPIC_BASE_URL=http://host.docker.internal:8081 \
  -e HOLODECK_PORT=8080 \
  -p 8080:8080 \
  holodeck/claude-assistant:latest
```

**Expected**: Agent runs as non-root, routes API calls through proxy, tmpfs provides scratch space.

## Scenario 4: Pre-Flight Validation Failure

```bash
# Without Node.js installed
holodeck serve agent.yaml
# Expected: Immediate error within 5 seconds:
# "Error: Node.js is required to run Claude Agent SDK but was not found on PATH."
# Server does NOT start.

# Without credentials
ANTHROPIC_API_KEY="" holodeck serve agent.yaml
# Expected: Immediate error:
# "Error: ANTHROPIC_API_KEY is not set. Set the environment variable or configure an alternative auth_provider."
```

## Scenario 5: Session Cap Under Load

```bash
# With max_concurrent_sessions: 5
# Send 6 concurrent requests
for i in $(seq 1 6); do
  curl -s -X POST http://localhost:8000/agent/claude-assistant/chat \
    -d '{"message": "Hello '$i'"}' &
done
wait
```

**Expected**: First 5 requests succeed. 6th returns HTTP 503 with `"error": "capacity_exceeded"`.
