# Agent Workflow Frameworks in Mid-2026: A Comparative Analysis

**Date:** 2026-07-25
**Purpose:** Comparative analysis of declarative/workflow capabilities across agent
frameworks, anchored on [Microsoft Agent Framework's declarative workflows](https://learn.microsoft.com/en-us/agent-framework/workflows/declarative?pivots=programming-language-python)
and the `workflows` module of `microsoft/agent-framework` — and a validation pass on
the thesis of [*Agent Workflows: A Solved Problem, Reinvented*](https://justinbarias.io/blog/agent-workflows-solved-problem-reinvented/)
(justinbarias.io, 2026-04-30).
**Related:** `deterministic-spine.md`, `specs/036-deterministic-spine/`

---

## 1. Microsoft Agent Framework: two workflow models under one brand

The single most important structural fact about MAF workflows is that **"declarative
workflows" and the code-first workflow engine are two different computational models**
that happen to share a runtime.

### 1.1 The code-first engine: Pregel/BSP, typed message passing

Source-verified facts (repo `microsoft/agent-framework`, `python/packages/core/agent_framework/_workflows/`, as of 2026-07-24):

- The `Workflow` docstring states it verbatim: *"A workflow executes a directed graph
  of executors connected via edge groups using a **Pregel-like model**, running in
  supersteps until the graph becomes idle."* `RunnerImpl` (`_runner.py`) is documented
  as running "Pregel supersteps."
- **Executors** declare capabilities via type annotations on `@handler` methods;
  `WorkflowContext[OutT, W_OutT]` is a generic capability token (send messages, yield
  outputs). Agents are auto-wrapped (`AgentExecutor`), and whole workflows compose as
  executors (`WorkflowExecutor`).
- **Edges** are typed and validated at `build()`: conditional edges,
  `SwitchCaseEdgeGroup` (first-match + default), fan-out (broadcast or
  `selection_func`), fan-in (join). Loops are cycles bounded by
  `max_iterations=100` supersteps (`WorkflowConvergenceException` beyond that).
- **BSP semantics**: messages produced in superstep *N* deliver at the start of
  *N+1*; shared `State` commits only at superstep boundaries; the runner docstring
  concedes "true parallelism is not realized in Python" (asyncio + GIL).
- **Checkpointing is superstep-granular**: `WorkflowCheckpoint` chains snapshots of
  in-flight messages, shared state, executor states, and pending
  human-input requests, keyed to a `graph_signature_hash` — resume refuses if the
  topology changed. Storage backends: in-memory, file, Cosmos DB.
- **Human-in-the-loop** is `ctx.request_info()` + `@response_handler` (the
  `RequestInfoExecutor` class was removed from Python in Nov 2025; some Learn pages
  are stale). Responses can be combined with checkpoint restore across process
  restarts.
- **Durability is layered, not native**: crash-safe replay comes from first-party
  bridges (`agent-framework-durabletask`, `agent-framework-azurefunctions`) that run
  MAF workflows *as Durable Task orchestrations* — executors mapped to activities,
  fan-out via `task_all`/`task_any`, HITL via external events.
- A second, **experimental "functional" API** (`@workflow`/`@step`, added 1.2.0,
  Apr 2026) lets you write workflows as plain async functions with native Python
  control flow — an implicit concession that the graph API is heavier than many
  tasks need.

Timeline: public preview Oct 1, 2025 (the AutoGen + Semantic Kernel convergence;
both predecessors in maintenance, SK's Process Framework explicitly discontinued);
heavy breaking-change churn through Feb 2026; **1.0 GA April 2, 2026**; current
release 1.12.1 (July 2026).

### 1.2 The declarative layer: Copilot Studio lineage, not the graph

The declarative YAML format ([Learn docs](https://learn.microsoft.com/en-us/agent-framework/workflows/declarative?pivots=programming-language-python),
doc dated 2026-06-26; **promoted RC → stable July 21, 2026** in core 1.12.0) is **not** a
serialization of the graph model. It is a **sequential action list**:

```yaml
actions:
  - kind: SetVariable
    variable: Local.greeting
    value: Hello
  - kind: If
    condition: =IsBlank(Local.AgentResult)
    then: [...]
  - kind: GotoAction        # jump-to-action-by-id — goto, in 2026
    actionId: retry_step
```

- **Action vocabulary**: `SetVariable`, `If`/`ConditionGroup`, `Foreach`,
  `BreakLoop`/`ContinueLoop`, `GotoAction`, `SendActivity`, `InvokeAzureAgent`,
  `InvokeFunctionTool`, `InvokeMcpTool`, `HttpRequestAction`, `Question`,
  `RequestExternalInput`, `EndWorkflow`/`EndConversation`. This taxonomy
  (`SendActivity`, `OnConversationStart`, `EndDialog`, `CancelAllDialogs`) is
  unmistakably **Bot Framework Adaptive Dialogs / Copilot Studio topic schema**
  lineage.
- **Expression language: Power Fx** — the Excel-formula language from the Power
  Platform, via the `powerfx` Python package. This is why **Python 3.14 is
  unsupported** ("due to PowerFx compatibility"). Not FEEL, not CEL, not any
  vendor-neutral standard.
- Under the hood, `WorkflowFactory` compiles each action into a real `Executor`
  node on the Pregel runtime (`IfConditionEvaluatorExecutor`,
  `ForeachInitExecutor`/`ForeachNextExecutor`, goto resolution with circular-goto
  validation) — so declarative workflows inherit checkpointing at action
  boundaries. An authoring layer, not a separate engine.
- **Language asymmetry**: the Python declarative docs show no checkpoint/resume
  surface at all (run-to-completion `workflow.run({...})`); CheckpointManager,
  superstep checkpoints, and resume-on-another-machine appear only in the C#
  pivot. Four conversation actions are C#-only. Python package
  `agent-framework-declarative` requires `--pre` per the current docs.
- **Error handling is manual**: the docs' recommended pattern is a
  `Local.hasError` flag plus `IsBlank()` checks after each agent call. No retry,
  compensation, or timeout semantics in the format.
- **Round-trip**: Microsoft Foundry's visual multi-agent Workflows designer
  (public preview, Ignite Nov 2025) is built on MAF and offers a synchronized
  visual ⇄ YAML dual view with VS Code export ("minimal changes" — hedged, not
  lossless). No verified round-trip with Copilot Studio topics.

**Assessment:** Microsoft did not adapt its graph engine for declarative authoring —
it grafted its low-code dialog format (Adaptive Dialogs + Power Fx) onto the agent
runtime. The result is two workflow models under one product name: a Pregel graph
for developers and a Power-Platform-style action list for the low-code audience,
with `GotoAction` as the escape hatch where a sequential list can't express a graph.

---

## 2. The landscape: who ships what (July 2026)

| Framework | Declarative format | Execution model | Expression language | Durability | Status (Jul 2026) |
|---|---|---|---|---|---|
| **MAF (code-first)** | — (code) | Pregel/BSP typed graph | none (Python predicates) | Superstep checkpoints; DurableTask bridge for real durability | 1.0 GA Apr 2026; 1.12.1 |
| **MAF (declarative)** | YAML action list (Adaptive Dialogs lineage) | Sequential actions + goto, compiled onto Pregel runtime | **Power Fx** | Inherited (C# surfaced; Python not documented) | Stable Jul 21, 2026 |
| **LangGraph** | **None official** — builder/YAML efforts archived Feb 2026; `langgraph.json` is a deploy manifest | Checkpointed state-machine graph | none (Python/TS callables) | Per-superstep checkpointers (SQLite/Postgres), time travel, `interrupt()` | 1.0 GA Oct 2025; 1.2.9 Jul 2026 |
| **CrewAI** | `agents.yaml`/`tasks.yaml` (content, not orchestration); Flows code-only | Crews: role-based; Flows: event-driven (`@start`/`@listen`/`@router`) | none | `@persist` → SQLite default; resume by state id | 1.0 GA Oct 2025; 1.15.6 Jul 2026 |
| **Google ADK** | Agent Config YAML (experimental, Gemini-only); Visual Builder emits it | 2.0 (May/Jun 2026): typed **graph engine** w/ dynamic nodes | none — "programmatic routing" is a stated design stance | First-party durable pause/resume, cross-runtime (Py↔Go); sessions in DB | 2.0 GA May 2026 (Py), Jun 2026 (Go) |
| **OpenAI Agents SDK / AgentKit** | **Agent Builder deprecated Jun 3, 2026** (shutdown Nov 30, 2026); export = lossy code, no spec | SDK: LLM loop + handoffs; Builder was node graph | Builder used **CEL** (dying with it); SDK none | Session stores only; **durability delegated to Temporal** (GA Mar 23, 2026) | Builder never left beta; SDK "model-native harness" pivot Apr 2026 |
| **AWS Bedrock Flows** | JSON nodes+connections, visual builder | Typed DAG, managed | JsonPath subset + custom 64-char predicates | Async executions ≤24h (preview) — managed, not replay | GA Nov 2024; quiet through 2026 |
| **AWS AgentCore** | None — framework-agnostic infra + Harness (API-configured) | Bring-your-own-framework; microVM sessions ≤8h | — | Session persistence, memory service | GA Oct 2025; classic Bedrock Agents → maintenance, closed to new customers Jul 30, 2026 |
| **AWS Strands** | None (code-first) | Graph/Swarm/handoff primitives in code | none | Via AgentCore pairing | 1.0 Jul 2025 |
| **Temporal** | None — workflows-as-code is the product | Durable execution (event-sourced replay) | none (host language) | The product itself | OpenAI Agents SDK integration GA **Mar 23, 2026** |
| **Dapr Workflow / Agents** | Components YAML; logic code-only | Durable execution on `durabletask-go` + actors | none | Event-history replay, 30+ state stores | Dapr Agents **v1.0 GA Mar 23, 2026** (KubeCon EU) |
| **Camunda 8 (BPMN/DMN)** | BPMN 2.0 XML + DMN decision tables | Token-flow engine; agents live inside **ad-hoc sub-processes** | **FEEL** (the DMN standard language, used engine-wide) | Engine-native: incidents, retries, audit, human tasks | AI Agent connector GA in 8.8 (Oct 2025); MCP server + A2A connectors in 8.9 (Apr 2026) |
| **Open Workflow Spec** (ex-CNCF Serverless Workflow) | YAML/JSON DSL 1.0 (Jan 2025) | Task-based, event-driven | **jq** (mandated default) | Runtime-dependent | Still CNCF Sandbox; renamed Nov 2025; MCP + A2A call tasks slated for v1.1; flagship runtime SonataFlow still implements spec 0.8 |
| **n8n** | JSON (nodes + connections), visual editor | Queue-mode (Redis + Postgres); AI Agent node is **LangChain.js** under the hood | `{{ single-line JS }}` (Tournament sandbox; two 2026 sandbox-escape CVEs) | Retry-from-failed-execution, not replay | $2.5B (Oct 2025) → **$5.2B after SAP stake (May 2026)** |

Notable cross-cutting facts:

- **Expression-language fragmentation is total.** Power Fx (Microsoft), CEL
  (OpenAI, being sunset), JSONata (AWS Step Functions), jq (Open Workflow Spec),
  sandboxed JavaScript (n8n), a 64-character custom predicate language (Bedrock
  Flows), or nothing at all (LangGraph, CrewAI, ADK, Strands, Temporal, Dapr).
  No agent framework adopted FEEL or any decision-logic standard — the only
  place FEEL meets agents in production is Camunda, where it was already the
  engine-wide expression language before the AI Agent connector arrived.
- **The word of 2026 is "harness," not "workflow."** OpenAI's April 2026 SDK
  evolution (model-native harness, sandbox, filesystem/shell tools, AGENTS.md),
  AWS AgentCore Harness (GA Jun 2026), and MAF's own Agent Harness (Build 2026)
  all converge on: give the model an environment and primitives, not a graph.
- **Durable execution had its coronation.** Temporal's OpenAI Agents SDK
  integration and Dapr Agents v1.0 went GA **on the same day** (March 23, 2026).
  Microsoft ships MAF-on-DurableTask bridges. The Diagrid position piece
  ("checkpoints are not durable execution") argues LangGraph/CrewAI/ADK-style
  checkpointing falls short for production — vendor-motivated but directionally
  aligned with the durable-execution camp.

### 2.1 The incumbents came to the agents

While agent frameworks were rebuilding workflow machinery, the workflow-engine
incumbents added agents to their existing standards stack — and their design is
telling:

- **Camunda 8.7 → 8.9 (Apr 2025 → Apr 2026)** made the BPMN 2.0 **ad-hoc
  sub-process** the "agent decision workspace": the LLM decides which tools to
  invoke and in what order *inside* a bounded container, while the engine
  provides retries, incident handling, audit logging, and human-task routing
  around it. The AI Agent connector went GA in 8.8 (Oct 2025); 8.9 added an
  Orchestration Cluster MCP server and A2A connectors. Determinism at the
  boundary, non-determinism inside a governed region — with FEEL as the
  expression layer throughout.
- **Flowable 2025.1 (Jul 2025)** shipped a first-class Agent Engine alongside
  its BPMN and CMMN engines, governed with the same tooling.
- **Open Workflow Specification** (the renamed CNCF Serverless Workflow) closed
  spec issues for **MCP and A2A call tasks** (Sep-Oct 2025, targeted at v1.1) —
  agents as just another protocol a standard workflow can call. Traction remains
  thin (still Sandbox after six years; reference runtime in alpha).

### 2.2 The commentary landscape

The "agents are reinventing workflow engines" debate matured considerably after
April 2026:

- **The durable-execution camp got loud and organized.** Temporal published a
  direct rebuttal to the "LLMs are non-deterministic, so Temporal doesn't fit"
  objection (*"Of course you can build dynamic AI agents with Temporal"*, Nov
  2025 — citing OpenAI Codex and Replit Agent 3 in production), and Restate,
  DBOS, and Inngest each published framework-agnostic durable-agent
  integrations. Every major cloud shipped durable primitives in the same window:
  AWS Lambda Durable Functions (Dec 2025), Cloudflare Workflows GA, Vercel
  Workflow DevKit, Azure Durable Task agent updates (Apr 2026).
- **The clearest independent statement of the thesis** is *"Agent Workflows Are
  Rediscovering Durable Execution"* (Koshy, May 18, 2026): agent systems are
  "unconsciously reinventing" Windows Workflow Foundation / Step Functions /
  Temporal lessons; the missing layer is "not another prettier graph — it is
  definitions, execution records, identity, policy, and replay"; what's needed
  is "a portable agent workflow definition, plus a durable execution contract."
- **Anthropic's position is unchanged since "Building Effective Agents" (Dec
  2024)**: composable patterns over frameworks, and through mid-2026 it still
  ships no workflow engine — the Claude Agent SDK is an agent loop with
  subagents, hooks, and skills as control surfaces, not a DAG runtime.
  **OpenAI's position collapsed into Anthropic's**: after the Agent Builder
  deprecation, its stack is code-first SDK + model-native harness + Temporal
  for durability.
- **The counter-current is commercial, not architectural**: n8n (visual
  workflows with an embedded LangChain.js agent node) more than doubled its
  valuation to $5.2B (SAP stake, May 2026) — evidence that "visual workflow +
  agent step" wins the mid-market automation buyer, even as its queue-and-retry
  execution model is exactly what the durable-execution camp critiques.
- **Where the debate settled**: a layering consensus. Durable execution won the
  reliability argument; model-led control flow won the orchestration argument
  (nobody serious argues for fully static DAGs anymore); the LLM decides inside
  a durable, replayable envelope. Camunda's ad-hoc sub-process and Temporal's
  "deterministic loop, non-deterministic decisions" are two industries
  independently converging on that same shape.

---

## 3. Verdict: does the blog's thesis hold?

The post made six load-bearing claims. Scored against the mid-2026 evidence:

### Claim-by-claim

1. **"Graph dataflow (BSP) is more machinery than typical agent apps need" — HOLDS,
   with one strong counterexample.** MAF's own trajectory concedes the point twice:
   the experimental functional API (plain async functions, Apr 2026) exists because
   the graph API is heavy, and the declarative layer abandons the graph model
   entirely for a sequential action list. LangChain archived its visual-builder and
   YAML-spec experiments (Feb 2026) rather than doubling down. OpenAI killed its
   node-graph builder. The counterexample is **Google ADK 2.0**, which replaced its
   simple Sequential/Parallel/Loop agents with a full typed graph engine in May-Jun
   2026 — one major vendor moved *toward* graph dataflow, not away.

2. **"Model labs deliberately don't bundle workflow engines" — STRONGLY CONFIRMED,
   now with a corpse.** Anthropic still ships no workflow engine (Claude Agent SDK
   remains loop + primitives). OpenAI went further than abstaining: it shipped a
   visual workflow builder (Oct 2025), watched it for eight months, **deprecated it
   June 3, 2026** (shutdown Nov 30, 2026), never GA'd it, and pivoted the Agents SDK
   to a "model-native harness" (Apr 2026). Its migration doc concedes workflows
   "with strong determinism at their core may not migrate faithfully." For crash-safe
   agents, OpenAI's answer is Temporal — exactly the "use a durable execution
   engine" prescription.

3. **"Four distinct orchestration categories; don't conflate them" — HOLDS.** The
   market sorted itself along precisely these lines: graph dataflow (MAF code-first,
   LangGraph, ADK 2.0), durable execution (Temporal, Dapr, DurableTask), and the
   categories didn't merge — they *bridged* (MAF-on-DurableTask, OpenAI-on-Temporal).

4. **"Durable execution is the most important category for agents" — CONFIRMED,
   and now conventional wisdom.** March 23, 2026: Temporal's OpenAI integration GA
   and Dapr Agents v1.0 GA on the same day. Microsoft, rather than making its
   Pregel engine natively durable, runs it *as* DurableTask orchestrations —
   vindicating the blog's specific jab that "Microsoft already ships DurableTask."
   Every major cloud shipped durable primitives within the year (Lambda Durable
   Functions, Cloudflare Workflows, Vercel Workflow DevKit). In-proc superstep
   checkpointing is increasingly called out (Diagrid, Restate, DBOS) as
   not-actually-durable-execution, and the Koshy essay (May 2026) independently
   restates the blog's core argument almost verbatim: agent systems are
   "unconsciously reinventing" lessons that WF, Step Functions, and Temporal
   already learned.

5. **"Models increasingly handle their own orchestration" — CONFIRMED AND NOW
   INDUSTRY DIRECTION.** The 2026 convergence on "harness" products (OpenAI Apr
   2026, AWS AgentCore Harness Jun 2026, MAF Agent Harness May 2026) is this claim
   productized: environment + tools + memory, control flow left to the model.

6. **"Bounded contexts / message buses are the overlooked architecture" — PARTIALLY
   VALIDATED.** No framework reoriented around bounded contexts, but A2A's
   trajectory (Linux Foundation donation Jun 2025, 150+ orgs, production use in year
   one) shows cross-context agent federation becoming real infrastructure, and
   Dapr Agents (pub/sub + actors per agent) is architecturally the closest thing to
   the blog's prescription shipping today.

### What the post undersold or got wrong

- **The "reinvention" is worse than claimed, in an unexpected direction.** The blog
  criticized frameworks for rebuilding *graph dataflow engines*. Microsoft's
  declarative layer actually rebuilt something older: a **2016-era bot dialog
  scripting format** (Adaptive Dialogs) with Excel formulas (Power Fx) and `goto`.
  The prediction that "XML-based BPM engines feel like overkill" has an ironic
  coda — the shipped alternative recreates pre-BPM scripting, minus the standards,
  the tooling, and the error-handling semantics (no retry/compensation/timeout;
  the documented pattern is a hand-rolled `hasError` flag).
- **ADK 2.0 is genuine counterevidence** to "heavy orchestration frameworks will
  feel like overkill." Google shipped *more* graph, *more* durability (first-party
  cross-runtime pause/resume), and made it GA. If the blog's trajectory argument is
  right, ADK 2.0 is the position that ages worst; it deserves acknowledgment as the
  strongest live counter-bet.
- **The lock-in argument gained a sharper form.** Agent Builder's death shows the
  failure mode isn't just learning-curve waste: OpenAI never exposed a portable
  declarative representation, so deprecation means lossy one-way code export.
  Frameworks with published, schema-validated formats (ADK's `AgentConfig.json`,
  MAF's YAML) at least leave artifacts behind. Portability of the *format*
  matters as much as maturity of the *engine*.
- **The mature-engines prescription found its proof point in Camunda, which the
  post didn't discuss.** "Use battle-tested orchestration primitives" turned out
  not to be hypothetical: Camunda demonstrated that a standard BPMN 2.0 element
  (the ad-hoc sub-process) can host an LLM's dynamic tool loop while the engine
  supplies retries, incidents, audit, and human tasks — no new workflow model
  required. That is the blog's argument implemented inside the standards stack.
- **The declarative mid-market is bigger than the post allowed.** n8n was filed
  under "declarative tools (for comparison)," but its $5.2B valuation (May 2026,
  SAP) says visual-workflow-with-an-agent-step is the commercially dominant form
  of agent orchestration outside engineering organizations. The thesis holds for
  engineers building products; the low-code buyer is a different market, and
  it's the market MAF's declarative layer and Bedrock Flows are actually aimed
  at.

### Overall

**The thesis holds — more strongly than when it was written.** Between April and
July 2026: OpenAI deprecated its workflow builder and standardized on
Temporal for durability; LangChain buried its declarative experiments; AWS put
classic Bedrock Agents into maintenance while Flows went quiet; the durable
execution camp (Temporal, Dapr, DurableTask) collected the wins; every vendor
launched a "harness"; and an independent essayist arrived at the same
"rediscovering durable execution" conclusion. The two data points running
against the thesis — ADK 2.0's graph engine and MAF declarative going stable —
are both first-party platform plays tied to visual builders and low-code
ecosystems (Vertex/Foundry), which is consistent with the blog's observation
that these engines serve the vendor's platform strategy more than the typical
agent developer.

One refinement the evidence suggests: the field didn't split into "frameworks
vs. mature tools" — it settled into a **layering consensus** the post only
partially anticipated. The model owns control flow inside a durable, replayable,
governed envelope; the envelope is supplied by durable-execution engines
(Temporal, Dapr, DurableTask) or standards-based engines (Camunda), not by the
agent framework's graph. The graph-dataflow layer specifically is the one whose
2026 obituaries (Agent Builder, langgraph-builder, Bedrock Flows' stall) keep
accumulating.

---

## 4. Implications for HoloDeck (036-deterministic-spine)

- **The FEEL/DMN choice is genuinely unoccupied ground among agent frameworks.**
  No agent framework adopted a standards-based decision language — the field
  split between proprietary (Power Fx), sunset (CEL in Agent Builder), jq/JS
  (workflow tools), and nothing (plain Python). The one ecosystem where FEEL
  meets agents — Camunda — is a heavyweight enterprise BPM platform, not a
  developer-first agent stack. A DMN-decision-table spine with FEEL is
  differentiated *and* standards-portable, which the Agent Builder shutdown
  shows is not a theoretical concern; and Camunda's success validates FEEL as
  proven at enterprise scale rather than an academic pick.
- **036 sits exactly in the gap the commentary names.** The Koshy essay's
  "what's missing" list — a portable agent workflow definition plus a durable
  execution contract, "definitions, execution records, identity, policy, and
  replay" — reads like the 036 spec's motivation section. The deterministic
  spine (versioned DMN tables, schema gates, replayable records, named human at
  the top) is a concrete instance of that missing layer, scoped to decisions
  rather than general orchestration.
- **Keep the spine thin.** The market is validating "deterministic decision
  logic + model-led leaves," not "workflow engine that owns the agents." The
  036 framing (LLM never the spine; agents at the edges behind schema gates)
  aligns with where the harness-era architecture is heading.
- **Publish the schema.** ADK's `AgentConfig.json` pattern (schema-validated,
  user-owned YAML) survived product churn; Agent Builder's internal-only graph
  format did not. 036's published JSON schema for decision tables is the right
  instinct — treat it as a contract, not an implementation detail.

---

*Research method: primary sources (repo source code of `microsoft/agent-framework`
at 2026-07-24, Microsoft Learn, official vendor blogs/changelogs/release notes,
PyPI) gathered July 25, 2026 via parallel research agents; facts are
source-verified unless framed as assessment. Key sources are linked inline
throughout.*
