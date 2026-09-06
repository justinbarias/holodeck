# Quickstart: Async Tool Initialization Endpoints

**Feature Branch**: `025-tool-init-endpoints`

## Prerequisites

- HoloDeck installed with `make install-dev`
- An agent config with at least one `vectorstore` or `hierarchical_document` tool
- Embedding provider credentials configured (e.g., `OPENAI_API_KEY` for OpenAI embeddings)
- For remote sources: install optional extras as needed:
  - S3: `pip install holodeck[s3]` (requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
  - Azure Blob: `pip install holodeck[azure-blob]` (requires `AZURE_STORAGE_CONNECTION_STRING`)
  - HTTP(S): no extra install needed

## Source Configuration Examples

```yaml
# Local path (current behavior, unchanged)
tools:
  - name: knowledge_base
    type: vectorstore
    source: data/docs/

# Volume mount (container deployments)
tools:
  - name: knowledge_base
    type: vectorstore
    source: /mnt/data/docs/

# S3 bucket
tools:
  - name: knowledge_base
    type: vectorstore
    source: s3://my-bucket/agents/customer-support/docs/

# Azure Blob Storage
tools:
  - name: knowledge_base
    type: vectorstore
    source: az://my-container/agents/docs/

# HTTP(S) single file
tools:
  - name: knowledge_base
    type: vectorstore
    source: https://cdn.example.com/datasets/knowledge-base.csv
```

## 1. Start the Server

```bash
holodeck serve agent.yaml --port 8000
```

## 2. List All Tools

```bash
curl -s http://localhost:8000/tools | python -m json.tool
```

Response:
```json
{
  "tools": [
    {
      "name": "knowledge_base",
      "type": "vectorstore",
      "supports_init": true,
      "init_status": null
    },
    {
      "name": "search_api",
      "type": "mcp",
      "supports_init": false,
      "init_status": null
    }
  ],
  "total": 2
}
```

## 3. Trigger Tool Initialization

```bash
curl -s -X POST http://localhost:8000/tools/knowledge_base/init \
  -w "\nHTTP Status: %{http_code}\nLocation: %{header_json}" \
  | python -m json.tool
```

Response (HTTP 201):
```json
{
  "tool_name": "knowledge_base",
  "state": "pending",
  "href": "/tools/knowledge_base/init",
  "created_at": "2026-03-23T10:00:00Z",
  "started_at": null,
  "completed_at": null,
  "message": "Initialization started",
  "error_detail": null,
  "progress": null,
  "force": false
}
```

## 4. Poll for Status

```bash
curl -s http://localhost:8000/tools/knowledge_base/init | python -m json.tool
```

Response (in progress):
```json
{
  "tool_name": "knowledge_base",
  "state": "in_progress",
  "href": "/tools/knowledge_base/init",
  "created_at": "2026-03-23T10:00:00Z",
  "started_at": "2026-03-23T10:00:01Z",
  "completed_at": null,
  "message": "Processing documents",
  "progress": {
    "documents_processed": 15,
    "total_documents": 42
  },
  "force": false
}
```

Response (completed):
```json
{
  "tool_name": "knowledge_base",
  "state": "completed",
  "href": "/tools/knowledge_base/init",
  "created_at": "2026-03-23T10:00:00Z",
  "started_at": "2026-03-23T10:00:01Z",
  "completed_at": "2026-03-23T10:00:45Z",
  "message": "Successfully initialized: 42 documents ingested",
  "progress": {
    "documents_processed": 42,
    "total_documents": 42
  },
  "force": false
}
```

## 5. Force Re-initialization

After updating source documents:

```bash
curl -s -X POST "http://localhost:8000/tools/knowledge_base/init?force=true" \
  | python -m json.tool
```

## 6. Error Cases

**Tool not found (404)**:
```bash
curl -s -X POST http://localhost:8000/tools/nonexistent/init
# {"detail": "Tool 'nonexistent' not found in agent configuration."}
```

**Unsupported tool type (400)**:
```bash
curl -s -X POST http://localhost:8000/tools/search_api/init
# {"detail": "Tool 'search_api' is of type 'mcp' which does not support initialization..."}
```

**Already initializing (409)**:
```bash
curl -s -X POST http://localhost:8000/tools/knowledge_base/init
# {"detail": "Tool 'knowledge_base' is currently being initialized..."}
```

**Concurrent limit reached (429)**:
```bash
# After triggering 3 concurrent inits (default limit)
curl -s -X POST http://localhost:8000/tools/another_tool/init
# {"detail": "Maximum concurrent initialization jobs (3) reached..."}
```

## Typical Deployment Flow

```bash
# 1. Start server
holodeck serve agent.yaml --port 8000 &

# 2. Wait for server readiness
until curl -sf http://localhost:8000/ready; do sleep 1; done

# 3. Pre-warm all vectorstore tools (check POST succeeded)
for tool in knowledge_base policy_docs; do
  status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:8000/tools/$tool/init")
  if [ "$status" != "201" ]; then
    echo "ERROR: POST /tools/$tool/init returned HTTP $status" && exit 1
  fi
  echo "Triggered init for $tool"
done

# 4. Wait for all tools to complete
for tool in knowledge_base policy_docs; do
  while true; do
    response=$(curl -s "http://localhost:8000/tools/$tool/init")
    state=$(echo "$response" | python -c "import sys,json; print(json.load(sys.stdin)['state'])")
    [ "$state" = "completed" ] && echo "$tool: ready" && break
    [ "$state" = "failed" ] && echo "FAILED: $tool" && echo "$response" && exit 1
    sleep 2
  done
done

# 5. Server is ready for chat traffic
echo "All tools pre-warmed. Ready for traffic."
```
