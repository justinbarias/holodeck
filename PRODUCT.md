# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Two people read HoloDeck's screens, often about the same agent:

- **Agent builders** (AI and platform engineers). They write the agent in YAML, run `holodeck test`, compare runs across prompt versions and models, and serve the agent. They are in the dashboard to find regressions and in the chat front end to try a change.
- **Domain reviewers** (for example policy analysts reviewing an agent that answers questions about legislation). They do not write YAML. They read answers, the sources behind them, and the date a fact held, and decide whether the agent is right.

## Product Purpose

HoloDeck is an open-source experimentation platform for AI agents. Teams define an agent, its tools and its evaluations in YAML, test it against structured cases, compare results, and deploy it as an HTTP service (REST or AG-UI), all from one CLI. Success means a team can show, with evidence, that an agent does the job before it ships: responses, tool use, evaluation scores, cost and latency.

## Positioning

The agent is a declarative YAML file, and every experiment produces saved evidence. The same definition runs on two first-class backends (Claude Agent SDK and OpenAI Agents), is graded by configurable evaluation models, and can answer questions about documents as of a date through temporal memory. The name comes from the Star Trek holodeck: a room where you run a scenario safely before it is real.

## Operating Context

- `holodeck test` writes timestamped result JSON under `results/<agent-slug>/`; `holodeck test view` serves a Dash dashboard (summary, explorer, compare) that reads those files and never executes an evaluation.
- `holodeck serve` exposes the agent over AG-UI; `sample/copilotkit` is the reference browser front end for any served agent. It renders tool calls inline, including memory views (fact timeline, as-of search results, provenance).
- Agents are document-heavy and multimodal: PDFs, Office files, images and text arrive as test inputs and chat attachments.
- Saved evidence and live execution are different things and must stay visibly distinct.

## Capabilities and Constraints

- Dashboard: Python Dash, styled by CSS assets in `src/holodeck/dashboard/assets/`. No JavaScript app.
- CopilotKit sample: Next.js 16, React 19, CopilotKit 1.87, AG-UI 1.0, Tailwind 4. Runtime configuration through environment variables (`AGENT_ID`, `AGENT_TITLE`, `AGENT_DESCRIPTION`, `HOLODECK_BACKEND_URL`, `FILE_ACCEPT`). It is generic: it must work for any HoloDeck agent, not only the legislation sample.
- The sample is a **forkable reference UI**: teams copy it as the starting point of a production front end, so it must read as a real product.
- Tool results arrive as JSON or plain strings; unknown tools fall back to a generic tool card.
- Terminology: agent, test case, run, metric, threshold, prompt version, tool call, fact, as-of date, compilation, provenance.

## Brand Commitments

- The name **HoloDeck** stays.
- The Star Trek holodeck idea stays: a simulated room for running scenarios before they are real.
- The current logo and the previous visual identity in `DESIGN.md` are not binding.

## Evidence on Hand

- Seeded dashboard fixture: `holodeck test view --seed` (24 runs, 6 prompt versions).
- Memory view fixtures: `sample/copilotkit/src/components/memory/fixtures/` and the `/memory-demo` page.
- A live sample over real legislation: `sample/temporal-memory` (Australian Family Assistance Acts, 2020 and 2026 compilations).
- There are no customer testimonials, logos, benchmarks or usage numbers. Do not invent them.

## Product Principles

1. Evidence first: every answer, score and fact shows where it came from and when it held.
2. One definition, many runs: the screens compare experiments; they do not hide differences between them.
3. Two readers, one screen: an engineer and a domain reviewer must both be able to act on what they see.
4. Saved is not live: results on disk and a running agent never look the same.
5. Generic by default: the reference UI works for any agent without code changes.

## Accessibility & Inclusion

WCAG 2.2 AA for text contrast, focus visibility and keyboard operation on both surfaces. Pass/fail and regressions must not rely on color alone.
