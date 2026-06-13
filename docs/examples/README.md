# Examples

Complete, runnable HoloDeck agents live in their own repository:

**👉 [github.com/justinbarias/holodeck-samples](https://github.com/justinbarias/holodeck-samples)**

Each sample ships for **OpenAI**, **Azure OpenAI**, and **Anthropic (Claude)**, so you can see the same agent run on either backend.

## Available samples

| Sample | What it shows |
|--------|---------------|
| [Ticket Routing](https://github.com/justinbarias/holodeck-samples/tree/main/ticket-routing) | Structured output, classification, confidence scoring |
| [Customer Support](https://github.com/justinbarias/holodeck-samples/tree/main/customer-support) | RAG, conversation memory, escalation workflows |
| [Content Moderation](https://github.com/justinbarias/holodeck-samples/tree/main/content-moderation) | Multi-category classification, policy enforcement |
| [Legal Summarization](https://github.com/justinbarias/holodeck-samples/tree/main/legal-summarization) | Document analysis, clause extraction, risk identification |
| [Legal Assistant](https://github.com/justinbarias/holodeck-samples/tree/main/legal-assistant) | Hierarchical document search, hybrid search, structured citations |
| [Financial Assistant](https://github.com/justinbarias/holodeck-samples/tree/main/financial-assistant) | Hierarchical document search, Qdrant native hybrid search, code graders, multi-turn test cases |

## Run one

```bash
git clone https://github.com/justinbarias/holodeck-samples
cd holodeck-samples/financial-assistant/openai   # or /claude, /azure
cp .env.sample .env        # fill in your credentials
holodeck test run agent.yaml
```

> The OpenAI/Azure samples need the [`openai-agents` extra](../getting-started/installation.md#backends-whats-included-vs-an-extra); RAG samples also need a vector-store extra (e.g. `qdrant`).

## Learn the building blocks

- [Agent Configuration](../guides/agent-configuration.md) — the full `agent.yaml` structure
- [Tools](../guides/tools.md) — function, vectorstore, hierarchical_document, MCP
- [Evaluations](../guides/evaluations.md) — test cases and metrics
- [OpenAI Backend](../guides/openai-backend.md) · [Claude Backend](../guides/claude-backend.md)
