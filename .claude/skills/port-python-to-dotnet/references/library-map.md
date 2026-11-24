# Library map: Python to .NET

A starting point for `port/DECISIONS.md`. It was put together in November 2026 from search results and vendor pages. Check each package on NuGet for its current version and status before you pin it, and use the Microsoft Learn MCP server for current API details.

Wrong library calls are the largest measured cause of failed translations, so settle every row before translation starts.

## Direct answers

| Python | .NET | Notes |
| --- | --- | --- |
| `pydantic` | Records with `System.Text.Json`, plus `Validate()` methods, DataAnnotations or FluentValidation | The serializer does not validate. See the JSON section of the rulebook |
| `click`, `typer`, `argparse` | `System.CommandLine` 2.x | Stable since November 2025. The API changed a lot from the long beta, so older examples are wrong. `Spectre.Console.Cli` is the other common choice |
| `rich` | `Spectre.Console` | Output will not match byte for byte. Decide what the judge compares |
| `fastapi`, `starlette` | ASP.NET Core minimal APIs | Server-sent events through `TypedResults.ServerSentEvents` on .NET 10 |
| `uvicorn` | Kestrel | Built in |
| `httpx`, `requests`, `aiohttp` | `HttpClient` from `IHttpClientFactory` | Never create one `HttpClient` per request |
| `pyyaml`, `ruamel.yaml` | `YamlDotNet` | Limited support for ahead-of-time compilation |
| `jinja2` | `Scriban` or `Fluid` | Neither is Jinja. Template syntax and whitespace rules differ, so templates need porting and judge cases |
| `jsonschema` | `JsonSchema.Net` | For validating against a schema. Use `JsonSchemaExporter` to produce one |
| `python-dotenv`, `os.environ` | `Microsoft.Extensions.Configuration` with environment variables | A `.env` loader needs a small package or a few lines of code |
| `logging`, `structlog` | `Microsoft.Extensions.Logging` with source-generated log methods | |
| `pytest` | xUnit v3 on Microsoft.Testing.Platform | Test filter syntax differs from the older runner |
| `pytest` snapshot plugins | `Verify` | |
| `hypothesis` | `CsCheck` or `FsCheck` | |
| `tenacity`, `backoff` | Polly through `Microsoft.Extensions.Resilience` | |
| `opentelemetry-*` | OpenTelemetry .NET | `System.Diagnostics.ActivitySource` is the tracing API |
| `temporalio` | `Temporalio` | Workflow code must not use `Task.Run`, `ConfigureAwait(false)`, timers or `DateTime.UtcNow`. Use the workflow's own APIs |
| `sqlalchemy` | EF Core, or Dapper for hand-written SQL | A large gap. Query behaviour needs its own judge cases |
| `numpy`, `pandas` | No like-for-like answer | Usually a reason to keep that part in Python |

## Model providers and agent libraries

| Python | .NET | Notes |
| --- | --- | --- |
| `anthropic` | `Anthropic` on NuGet | The official package from version 10. `Anthropic.SDK` and `tryAGI.Anthropic` are community packages |
| `openai` | `OpenAI` on NuGet | Official. `Azure.AI.OpenAI` adds Azure support on top |
| `ollama` | `OllamaSharp` | |
| `mcp` | `ModelContextProtocol` | Official C# SDK, 1.0 in March 2026. Supports stdio and HTTP on both client and server |
| A provider abstraction of your own | `Microsoft.Extensions.AI` and its `IChatClient` interface | The provider packages above plug into it. Middleware covers telemetry, caching and tool calls |
| Vector store clients | `Microsoft.Extensions.VectorData` with a connector per store | Several connectors were still in preview. Check each store you use |
| `claude-agent-sdk` | No official .NET SDK was found | A gap. See below |
| `openai-agents` | No official .NET SDK was found | A gap. See below |
| `semantic-kernel`, `autogen` | Microsoft Agent Framework | 1.0 in April 2026 for Python and .NET. Its Anthropic provider was still prerelease |

## When there is no .NET version

The research found no official .NET version of the Claude Agent SDK or the OpenAI Agents SDK. Check again before deciding, because that finding is an absence of search results and may be out of date. If it still holds, the user picks one of three answers and it goes in `port/DECISIONS.md`.

Keeping that part in Python is the lowest risk. The .NET build talks to it over HTTP, stdio or MCP, and the port can ship without it. Rebuilding the agent loop on `Anthropic` or `OpenAI` with `IChatClient` and MCP tools gives a single runtime, but the loop's behaviour then needs its own judge cases. Community packages that wrap a vendor CLI exist and describe themselves as unofficial, so check their upkeep and licence first.

## Interop, if you need it

| Option | Use it for | Limits |
| --- | --- | --- |
| A separate Python process over HTTP, stdio or MCP | Keeping a backend in Python, and running Python as the judge's reference | Startup time and a second runtime to deploy |
| CSnakes | Calling typed Python functions inside the .NET process | Embeds CPython, so the global interpreter lock applies. Check its supported .NET and Python versions |
| Python.NET | Dynamic access to Python objects inside the process | Calls need the lock and objects are `dynamic`, which the rulebook bans in product code |
| IronPython | Nothing in a modern port | Implements Python 3.4 and cannot load C extensions |
