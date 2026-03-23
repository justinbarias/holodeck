# Contract: Session Capacity Error

**Trigger**: Request arrives when `active_sessions >= max_concurrent_sessions`
**Protocol**: REST and AG-UI

## REST Response

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 5
Content-Type: application/json

{
  "error": "capacity_exceeded",
  "message": "Maximum concurrent Claude sessions (10) reached. Retry after existing sessions complete.",
  "active_sessions": 10,
  "max_sessions": 10
}
```

## AG-UI Response

Server-Sent Event with error type:
```
event: error
data: {"type": "capacity_exceeded", "message": "Maximum concurrent Claude sessions (10) reached."}
```

## Subprocess Crash Response

**Trigger**: Claude SDK subprocess terminates unexpectedly mid-session

### REST
```http
HTTP/1.1 502 Bad Gateway
Content-Type: application/json

{
  "error": "backend_error",
  "message": "Claude Agent SDK subprocess terminated unexpectedly. Start a new session to retry.",
  "retriable": true
}
```

### AG-UI
```
event: error
data: {"type": "backend_error", "message": "Claude Agent SDK subprocess terminated unexpectedly.", "retriable": true}
```
