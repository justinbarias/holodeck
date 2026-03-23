# Contract: Enhanced Health Endpoint

**Endpoint**: `GET /health`
**Change type**: Backward-compatible extension

## Current Response (unchanged fields)

```json
{
  "status": "healthy | unhealthy",
  "agent_name": "string | null",
  "agent_ready": true,
  "active_sessions": 0,
  "uptime_seconds": 0.0
}
```

## Extended Response (new fields)

```json
{
  "status": "healthy | degraded | unhealthy",
  "agent_name": "string | null",
  "agent_ready": true,
  "active_sessions": 0,
  "uptime_seconds": 0.0,
  "backend_ready": true,
  "backend_diagnostics": []
}
```

## Status Values

| Status | Meaning |
|--------|---------|
| `healthy` | Server running, backend prerequisites met |
| `degraded` | Server running, backend prerequisites have warnings |
| `unhealthy` | Server running, backend prerequisites failed |

## Backward Compatibility

- New fields are additive; existing consumers ignore unknown fields
- `status` value `"degraded"` is new; consumers checking `status == "healthy"` will correctly treat degraded as not-healthy
- `agent_ready` semantics unchanged
