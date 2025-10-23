# Frequently Asked Questions

This is a sample FAQ file for your agent. You can:
- Add your own FAQs here
- Use this as a vector search data source
- Reference it in your agent instructions

## General Questions

### What is this agent?

This agent is a conversational AI assistant designed to help users through natural dialogue.

### How do I ask questions?

Simply type your question or statement, and the agent will respond based on its training and configuration.

### What can this agent do?

This agent can:
- Answer questions on various topics
- Engage in multi-turn conversations
- Remember context within a conversation
- Use tools (once configured) to extend functionality

### What are the limitations?

The agent:
- Has a knowledge cutoff date
- Cannot access real-time information without tools
- May not be perfect on all topics
- Should be verified for critical applications

## Technical Questions

### How do I add tools?

See `../tools/README.md` for detailed instructions on adding tools.

### How do I test the agent?

Run: `holodeck test agent.yaml`

Test cases are embedded in the `agent.yaml` file under the `test_cases` field.

### How do I customize the system prompt?

Edit `instructions/system-prompt.md` to change the agent's behavior and tone.

### How do I change the model?

Edit `agent.yaml` and modify the `model` section with your preferred provider and model name.

## Troubleshooting

### Agent is giving incorrect responses

- Check your system prompt in `instructions/system-prompt.md`
- Verify tool configurations in `agent.yaml`
- Review test cases in the `test_cases` field of `agent.yaml`

### Changes aren't taking effect

- Save all files
- Restart the agent process
- Check for syntax errors in `agent.yaml`

### How do I get help?

Visit the [HoloDeck documentation](https://docs.holodeck.dev) or check the main README in the parent directory.
