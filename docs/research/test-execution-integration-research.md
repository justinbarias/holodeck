# Test Execution Framework Integration Research

**Date**: November 1, 2025
**Purpose**: Research best practices for integrating key technologies into HoloDeck's Python CLI test execution framework
**Scope**: Semantic Kernel Agents, markitdown, Azure AI Evaluation SDK, and NLP metrics libraries

---

## Table of Contents

1. [Semantic Kernel Agents](#1-semantic-kernel-agents)
2. [markitdown - Document Conversion](#2-markitdown---document-conversion)
3. [Azure AI Evaluation SDK](#3-azure-ai-evaluation-sdk)
4. [NLP Metrics Libraries](#4-nlp-metrics-libraries)
5. [Integration Architecture Recommendations](#5-integration-architecture-recommendations)
6. [Decision Matrix](#6-decision-matrix)

---

## 1. Semantic Kernel Agents

### Overview

Semantic Kernel is Microsoft's open-source SDK for orchestrating AI services, providing a framework for building AI agents with tool calling, memory management, and multi-agent orchestration.

### Installation & Setup

```bash
# Basic installation
pip install semantic-kernel

# With specific integrations
pip install semantic-kernel[azure]
pip install semantic-kernel[hugging-face]

# Install all optional dependencies
pip install semantic-kernel[all]
```

**Requirements**:
- Python >= 3.10 (3.10, 3.11, 3.12 supported)
- Latest version: 1.37.1 (as of October 30, 2025)
- License: MIT

**Optional Extras**: anthropic, autogen, aws, azure, chroma, copilotstudio, dapr, faiss, google, hugging-face, mcp, milvus, mistralai, mongo, notebooks, ollama, onnx, pandas, pinecone, postgres, qdrant, realtime, redis, sql, usearch, weaviate

### Agent Invocation Patterns

#### Basic Agent Invocation

Semantic Kernel supports two primary non-streaming invocation methods:

```python
from semantic_kernel.agents import Agent

# Method 1: Invoke with messages (positional argument)
response = await agent.invoke(messages)

# Method 2: Invoke with messages (keyword argument)
response = await agent.invoke(messages=messages)

# Method 3: Invoke without messages (agent uses instructions only)
response = await agent.invoke()
```

**Key Convention**: All arguments except the first positional `messages` parameter must be passed as keyword arguments.

#### Orchestration Patterns (2025)

All orchestration patterns share a unified interface:

```python
from semantic_kernel.orchestration import SequentialOrchestration, InProcessRuntime

# Create orchestration
orchestration = SequentialOrchestration(members=[agent_a, agent_b])

# Start runtime
runtime = InProcessRuntime()
runtime.start()

# Invoke with task
result = await orchestration.invoke(task="Your task here", runtime=runtime)

# Get final output
final_output = await result.get()

# Cleanup
await runtime.stop_when_idle()
```

**Available Patterns**: Concurrent, Sequential, Handoff, Group Chat, Magentic

#### Timeout Configuration

Developers can specify a timeout when retrieving orchestration results:

```python
# With timeout
try:
    result = await orchestration_result.get(timeout=30.0)  # 30 seconds
except TimeoutException:
    # Handle timeout
    pass
```

### Passing Context & Files to Agents

#### Chat History Management

```python
from semantic_kernel.contents import ChatHistory

# Create chat history
history = ChatHistory()

# Add messages
history.add_system_message("You are a helpful assistant")
history.add_user_message("What is the capital of France?")
history.add_assistant_message("The capital of France is Paris.")

# Pass to agent
response = await agent.invoke(messages=history)
```

#### File Context Integration

For test scenarios requiring file inputs:

```python
from semantic_kernel.contents import ChatHistory, ChatMessageContent

# Create message with file context
history = ChatHistory()

# Add file content as context
file_context = f"File: {filename}\n\n{file_content}"
history.add_user_message(f"Analyze this document:\n\n{file_context}")

# Invoke agent
response = await agent.invoke(messages=history)
```

#### Thread-Based State Management

```python
from semantic_kernel.agents import ChatHistoryAgentThread

# Create or retrieve thread for user/session
thread = ChatHistoryAgentThread()

# Add message to thread
await thread.add_message(user_message)

# Get response with maintained state
response = await agent.get_response(thread=thread)
```

### Capturing Agent Responses & Tool Calls

#### Response Structure

```python
from semantic_kernel.agents import Agent

# Invoke agent
response = await agent.invoke(messages=history)

# Access response content
response_text = response.content

# Access metadata
metadata = response.metadata

# Check for tool calls
if hasattr(response, 'tool_calls'):
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")
```

#### Capturing Tool Call Details

```python
from semantic_kernel.contents import ChatMessageContent

async def invoke_agent_with_tool_tracking(agent, history):
    """Invoke agent and track tool calls"""
    tool_calls = []

    # Invoke agent
    response = await agent.invoke(messages=history)

    # Extract tool calls from response
    for message in response.messages:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append({
                    'name': tool_call.function.name,
                    'arguments': tool_call.function.arguments,
                    'id': tool_call.id
                })

    return {
        'response': response.content,
        'tool_calls': tool_calls,
        'metadata': response.metadata
    }
```

### Error Handling & Timeouts

#### HTTP Client Configuration

```python
import httpx
from semantic_kernel import Kernel

# Configure custom HTTP client with timeouts
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=10.0,
        pool=5.0
    )
)

# Create kernel with custom HTTP client
kernel = Kernel(http_client=http_client)
```

#### Built-in Retry Policy

Semantic Kernel with Azure OpenAI includes:
- Automatic retry up to 3 times
- Exponential backoff
- Detection of `retry-after` HTTP headers for tailored retries

#### Custom Retry Logic

```python
from semantic_kernel.exceptions import ServiceException
import asyncio

async def invoke_with_retry(agent, messages, max_retries=3, backoff_factor=2):
    """Invoke agent with custom retry logic"""
    for attempt in range(max_retries):
        try:
            return await agent.invoke(messages=messages)
        except ServiceException as e:
            if attempt == max_retries - 1:
                raise

            wait_time = backoff_factor ** attempt
            await asyncio.sleep(wait_time)

            # Log retry attempt
            print(f"Retry attempt {attempt + 1} after {wait_time}s")
```

#### Tool Call Error Handling

```python
from semantic_kernel.functions import KernelFunction

async def safe_function_call(function: KernelFunction, **kwargs):
    """Execute function with error handling"""
    try:
        return await function.invoke(**kwargs)
    except Exception as e:
        # Return error information to LLM
        return {
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Tool execution failed'
        }
```

### Best Practices

1. **State Management**
   - Use `ChatHistoryAgentThread` for session-based conversations
   - Serialize chat history to Cosmos DB or similar for persistence
   - Use history reducers (`ChatHistorySummarizationReducer`, `ChatHistoryTruncationReducer`) for long conversations

2. **Timeout Configuration**
   - Set HTTP client timeouts at the kernel level
   - Configure operation-level timeouts for orchestration
   - Default timeout is ~100 seconds; adjust based on agent complexity

3. **Error Handling**
   - Use built-in retry mechanisms when possible
   - Implement custom retry logic only for specific requirements
   - Capture and log all tool call errors
   - Return error context to LLM for self-correction

4. **Context Management**
   - Use `auto_reduce` with history reducers to maintain context window limits
   - Configure `target_count` and `threshold_count` appropriately
   - Preserve critical message pairs during reduction

5. **Tool Invocation**
   - Configure function timeouts at the HTTP client level
   - Implement graceful degradation for tool failures
   - Pass tool errors back to the model for retry attempts

### Common Pitfalls

1. **Timeout Confusion**: Timeouts can occur at multiple layers (gRPC, HTTP client, Semantic Kernel). Always check which layer is timing out.

2. **State Leakage**: Some agents require full chat history on each invocation. Ensure proper state management strategy.

3. **Tool Call Failures**: Without proper error handling, tool failures can crash the agent. Always implement try-catch patterns.

4. **Nested Retry Logic**: Don't stack retry mechanisms at multiple layers. Use built-in retries or implement at one level only.

5. **History Bloat**: Long conversations can exceed context windows. Use history reducers proactively.

### Integration Considerations

For HoloDeck's test execution framework:

1. **Test Case Execution**
   ```python
   async def execute_test_case(agent, test_case):
       """Execute a single test case"""
       # Create chat history
       history = ChatHistory()
       history.add_system_message(agent.instructions)

       # Add test input
       history.add_user_message(test_case.input)

       # Invoke agent with timeout
       try:
           response = await invoke_with_retry(
               agent,
               history,
               max_retries=3
           )

           return {
               'output': response.content,
               'tool_calls': extract_tool_calls(response),
               'status': 'success'
           }
       except Exception as e:
           return {
               'output': None,
               'error': str(e),
               'status': 'error'
           }
   ```

2. **Multi-Turn Conversations**
   ```python
   async def execute_multi_turn_test(agent, test_case):
       """Execute test with multiple turns"""
       thread = ChatHistoryAgentThread()
       results = []

       for turn in test_case.turns:
           await thread.add_message(turn.input)
           response = await agent.get_response(thread=thread)

           results.append({
               'input': turn.input,
               'output': response.content,
               'tool_calls': extract_tool_calls(response)
           })

       return results
   ```

3. **File Input Handling**
   ```python
   async def execute_test_with_files(agent, test_case):
       """Execute test with file inputs"""
       history = ChatHistory()

       # Load and convert files
       for file_input in test_case.files:
           file_content = await load_file(file_input)
           history.add_user_message(
               f"File: {file_input.path}\n\n{file_content}"
           )

       # Add test query
       history.add_user_message(test_case.query)

       # Invoke agent
       response = await agent.invoke(messages=history)
       return response
   ```

---

## 2. markitdown - Document Conversion

### Overview

MarkItDown is Microsoft's open-source Python utility for converting various file formats to Markdown for use with LLMs and text analysis pipelines. It has gained over 25k GitHub stars since its debut.

### Installation & Setup

```bash
# Basic installation
pip install markitdown

# With all optional dependencies (recommended)
pip install 'markitdown[all]'
```

**Requirements**:
- Python >= 3.8 (recommended 3.10+)
- License: MIT

**Optional Dependencies**:
- Vision models for image descriptions (OpenAI, Azure OpenAI)
- OCR libraries for image text extraction
- Audio transcription services

### Supported File Types

- **Office Documents**: DOCX, XLSX, PPTX
- **Documents**: PDF, TXT, RTF
- **Web**: HTML, XML
- **Data**: JSON, CSV
- **Media**: Images (JPG, PNG, GIF with OCR), Audio (with transcription)
- **Archives**: ZIP (with recursive processing)

### API Usage Patterns

#### Basic Conversion

```python
from markitdown import MarkItDown

# Create instance
md = MarkItDown()

# Convert file
result = md.convert("document.pdf")

# Access markdown content
markdown_text = result.text_content

# Access metadata
title = result.title
```

#### Stream Processing

```python
from markitdown import MarkItDown
import io

md = MarkItDown()

# Convert from file-like object (binary mode required)
with open("document.pdf", "rb") as f:
    result = md.convert_stream(f)
    print(result.text_content)

# Convert from BytesIO
file_bytes = io.BytesIO(binary_data)
result = md.convert_stream(file_bytes)
```

#### LLM Integration for Image Descriptions

```python
from markitdown import MarkItDown
from openai import OpenAI

# Create OpenAI client
client = OpenAI(api_key="your-api-key")

# Create MarkItDown with LLM
md = MarkItDown(llm_client=client, llm_model="gpt-4o")

# Convert image with AI-generated description
result = md.convert("image.jpg")
print(result.text_content)  # Includes AI-generated description
```

#### Plugin Support

```python
from markitdown import MarkItDown

# Enable plugins
md = MarkItDown(enable_plugins=True)

result = md.convert("document.pdf")
```

### Handling Large Files

#### Memory Considerations

```python
from markitdown import MarkItDown
import tempfile

async def convert_large_file(file_path, chunk_size=1024*1024):
    """Convert large file with streaming"""
    md = MarkItDown()

    # For very large files, consider processing in chunks
    with open(file_path, "rb") as f:
        result = md.convert_stream(f)
        return result.text_content
```

#### Temporary File Storage

```python
import tempfile
from pathlib import Path

async def convert_with_temp_storage(file_stream, filename):
    """Convert using temporary file"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        # Write stream to temp file
        tmp.write(file_stream.read())
        tmp_path = tmp.name

    try:
        # Convert from temp file
        md = MarkItDown()
        result = md.convert(tmp_path)
        return result.text_content
    finally:
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
```

#### Chunked Processing for Large Archives

```python
from markitdown import MarkItDown

async def convert_large_pst(pst_path, max_attachments=None):
    """Convert large PST with attachment filtering"""
    md = MarkItDown()

    # Configure filtering to skip unnecessary attachments
    # Note: Actual filtering would depend on PST processing implementation
    result = md.convert(pst_path)

    return result.text_content
```

### Error Handling for Corrupted/Malformed Files

#### Unicode Encoding Errors

```python
from markitdown import MarkItDown
from markitdown._markitdown import UnsupportedFormatException

def safe_convert(file_path):
    """Convert file with error handling"""
    md = MarkItDown()

    try:
        result = md.convert(file_path)
        return result.text_content
    except UnicodeEncodeError as e:
        # Handle encoding issues
        print(f"Encoding error: {e}")
        # Try with different encoding or skip problematic characters
        return None
    except UnsupportedFormatException as e:
        print(f"Unsupported format: {e}")
        return None
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return None
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None
```

#### Robust Conversion with Fallbacks

```python
from markitdown import MarkItDown

def convert_with_fallback(file_path):
    """Convert with fallback strategies"""
    md = MarkItDown()

    try:
        # Try basic conversion
        result = md.convert(file_path)
        return result.text_content
    except Exception as e:
        print(f"Basic conversion failed: {e}")

        try:
            # Try with plugins
            md_with_plugins = MarkItDown(enable_plugins=True)
            result = md_with_plugins.convert(file_path)
            return result.text_content
        except Exception as e2:
            print(f"Plugin conversion failed: {e2}")

            # Last resort: return error marker
            return f"[CONVERSION_FAILED: {file_path}]"
```

#### File Validation

```python
from pathlib import Path
import magic  # python-magic library

def validate_file_before_conversion(file_path):
    """Validate file before attempting conversion"""
    path = Path(file_path)

    # Check existence
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check size (e.g., warn if > 100MB)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 100:
        print(f"Warning: Large file detected ({size_mb:.2f}MB)")

    # Check file type
    try:
        mime = magic.from_file(str(path), mime=True)
        print(f"Detected MIME type: {mime}")
    except Exception as e:
        print(f"Could not detect MIME type: {e}")

    return True
```

### Best Practices

1. **File Type Detection**
   - Always validate file type before conversion
   - Use MIME type detection for uploaded files
   - Handle unsupported formats gracefully

2. **Memory Management**
   - Use streaming for files > 10MB
   - Consider temporary file storage for network uploads
   - Implement cleanup for temporary files

3. **Error Recovery**
   - Implement multi-tier fallback strategy
   - Log conversion failures for debugging
   - Return meaningful error messages

4. **Performance Optimization**
   - Cache converted results when possible
   - Use async I/O for multiple files
   - Filter unnecessary content (e.g., PST attachments)

5. **LLM Integration**
   - Use vision models only when image descriptions are needed
   - Consider cost implications of LLM calls
   - Implement rate limiting for API calls

### Common Pitfalls

1. **Unicode Encoding**: PDF files often contain special characters that cause encoding errors. Use error='replace' or 'ignore' modes.

2. **Network Paths**: Conversion fails on mapped network drives. Copy to local filesystem first.

3. **Large Files**: Memory issues with files > 100MB. Use streaming or chunked processing.

4. **Binary vs Text Mode**: `convert_stream()` requires binary file-like objects, not text mode.

5. **Temporary File Cleanup**: Forgetting to delete temporary files leads to disk bloat.

### Integration Considerations

For HoloDeck's test execution framework:

1. **Test File Preprocessing**
   ```python
   from markitdown import MarkItDown

   async def preprocess_test_files(test_case):
       """Convert all test case files to markdown"""
       md = MarkItDown()
       converted_files = []

       for file_input in test_case.files:
           try:
               # Validate file
               validate_file_before_conversion(file_input.path)

               # Convert to markdown
               result = md.convert(file_input.path)

               converted_files.append({
                   'original_path': file_input.path,
                   'markdown_content': result.text_content,
                   'title': result.title,
                   'status': 'success'
               })
           except Exception as e:
               converted_files.append({
                   'original_path': file_input.path,
                   'error': str(e),
                   'status': 'error'
               })

       return converted_files
   ```

2. **Multimodal Test Support**
   ```python
   async def prepare_multimodal_context(test_case):
       """Prepare context from multiple file types"""
       md = MarkItDown()
       context_parts = []

       for file_input in test_case.files:
           file_type = file_input.type  # image, pdf, excel, etc.

           if file_type == 'image':
               # Use LLM for image description
               client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
               md_with_vision = MarkItDown(llm_client=client, llm_model="gpt-4o")
               result = md_with_vision.convert(file_input.path)
           else:
               # Standard conversion
               result = md.convert(file_input.path)

           context_parts.append(f"## {file_input.path}\n\n{result.text_content}")

       return "\n\n".join(context_parts)
   ```

3. **Excel Sheet/Range Selection**
   ```python
   async def convert_excel_with_selection(file_path, sheet=None, range=None):
       """Convert Excel with specific sheet/range"""
       # Note: markitdown may not support range selection natively
       # Would need to use openpyxl or pandas for preprocessing

       if sheet or range:
           import pandas as pd

           # Read specific sheet/range
           df = pd.read_excel(
               file_path,
               sheet_name=sheet,
               usecols=range
           )

           # Convert to markdown table
           markdown_table = df.to_markdown(index=False)
           return markdown_table
       else:
           # Use markitdown for full file
           md = MarkItDown()
           result = md.convert(file_path)
           return result.text_content
   ```

4. **PDF Page Range Selection**
   ```python
   import PyPDF2
   from markitdown import MarkItDown

   async def convert_pdf_pages(file_path, start_page=None, end_page=None):
       """Convert specific PDF pages"""
       if start_page is not None or end_page is not None:
           # Extract specific pages
           with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
               reader = PyPDF2.PdfReader(file_path)
               writer = PyPDF2.PdfWriter()

               start = start_page or 0
               end = end_page or len(reader.pages)

               for page_num in range(start, end):
                   writer.add_page(reader.pages[page_num])

               writer.write(tmp)
               tmp_path = tmp.name

           try:
               md = MarkItDown()
               result = md.convert(tmp_path)
               return result.text_content
           finally:
               Path(tmp_path).unlink(missing_ok=True)
       else:
           # Convert full PDF
           md = MarkItDown()
           result = md.convert(file_path)
           return result.text_content
   ```

---

## 3. Azure AI Evaluation SDK

### Overview

Azure AI Evaluation SDK provides built-in evaluators for assessing generative AI applications, including generation quality metrics (groundedness, relevance, coherence, fluency) and safety metrics.

### Installation & Setup

```bash
# Install Azure AI Evaluation SDK
pip install azure-ai-evaluation
```

**Requirements**:
- Python >= 3.9
- Azure AI Foundry Project or Azure OpenAI (for AI-assisted evaluators)

**Recent Changes**:
- Removed numpy dependency (now uses `math.nan` instead of `numpy.nan`)
- Removed `[remote]` extra (no longer needed for Azure AI Foundry portal tracking)
- Removed `azure-ai-inference` dependency
- `credential` now required for content safety evaluators and ProtectedMaterialsEvaluator

### Available Built-in Metrics

#### AI-Assisted Quality Evaluators

These require LLM access (GPT-3.5-turbo, GPT-4, GPT-4-turbo, GPT-4o, or GPT-4o-mini):

- **GroundednessEvaluator**: Assesses correspondence between claims in AI-generated answers and source context
- **RelevanceEvaluator**: Measures relevance of response to query
- **CoherenceEvaluator**: Evaluates logical flow and readability
- **FluencyEvaluator**: Assesses language quality
- **SimilarityEvaluator**: Compares semantic similarity
- **RetrievalEvaluator**: Evaluates retrieval quality

#### Agent-Specific Evaluators

- **IntentResolution**: Evaluates if agent resolved user intent
- **ToolCallAccuracy**: Assesses accuracy of tool calls
- **TaskAdherence**: Measures adherence to task requirements

#### Token Limits

- Most evaluators: 800 max_tokens
- RetrievalEvaluator: 1600 max_tokens
- ToolCallAccuracyEvaluator: 3000 max_tokens

### API Usage Patterns

#### Basic Evaluation

```python
from azure.ai.evaluation import evaluate, GroundednessEvaluator, RelevanceEvaluator
from azure.ai.evaluation import AzureOpenAIModelConfiguration

# Configure model
model_config = AzureOpenAIModelConfiguration(
    azure_endpoint="https://your-endpoint.openai.azure.com",
    api_key="your-api-key",
    azure_deployment="gpt-4o"
)

# Create evaluators
groundedness = GroundednessEvaluator(model_config=model_config)
relevance = RelevanceEvaluator(model_config=model_config)

# Evaluate single response
result = groundedness(
    query="What is the capital of France?",
    context="France is a country in Europe. Paris is its capital city.",
    response="The capital of France is Paris."
)

print(f"Groundedness Score: {result['groundedness']}")
```

#### Batch Evaluation

```python
from azure.ai.evaluation import evaluate

# Prepare data
data = [
    {
        "query": "What is the capital of France?",
        "context": "France is a country in Europe. Paris is its capital city.",
        "response": "The capital of France is Paris."
    },
    {
        "query": "What is machine learning?",
        "context": "Machine learning is a subset of AI...",
        "response": "Machine learning enables computers to learn..."
    }
]

# Run evaluation
results = evaluate(
    data=data,
    evaluators={
        "groundedness": groundedness,
        "relevance": relevance
    },
    evaluator_config={
        "groundedness": {
            "column_mapping": {
                "query": "query",
                "context": "context",
                "response": "response"
            }
        }
    }
)

print(results)
```

#### Query as Optional Input (New Feature)

```python
# GroundednessEvaluator now supports query as optional input
result = groundedness(
    query="What is the capital of France?",  # Optional but recommended
    context="France is a country in Europe. Paris is its capital city.",
    response="The capital of France is Paris."
)

# If query is provided, a different prompt template is used
```

### Model Configuration (Per-Metric Overrides)

#### Global Model Configuration

```python
from azure.ai.evaluation import AzureOpenAIModelConfiguration

# Global model config (used by all evaluators by default)
global_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="gpt-4o-mini"  # Cost-effective option
)
```

#### Per-Evaluator Model Override

```python
# Use different models for different evaluators
expensive_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="gpt-4o"  # More expensive, higher quality
)

cheap_config = AzureOpenAIModelConfiguration(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="gpt-4o-mini"  # Less expensive
)

# Critical evaluators use expensive model
groundedness = GroundednessEvaluator(model_config=expensive_config)
relevance = RelevanceEvaluator(model_config=expensive_config)

# Less critical evaluators use cheap model
fluency = FluencyEvaluator(model_config=cheap_config)
coherence = CoherenceEvaluator(model_config=cheap_config)
```

#### Per-Metric Parameters

```python
# Configure max_tokens per evaluator
groundedness = GroundednessEvaluator(
    model_config=model_config,
    max_tokens=800
)

retrieval = RetrievalEvaluator(
    model_config=model_config,
    max_tokens=1600  # Higher limit for retrieval
)
```

### Custom Metric Implementation

#### Basic Custom Evaluator

```python
from promptflow.core import Prompty
from promptflow.core import AzureOpenAIModelConfiguration
import os

class FriendlinessEvaluator:
    """Custom evaluator for measuring friendliness"""

    def __init__(self, configuration: AzureOpenAIModelConfiguration):
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, "friendliness.prompty")

        # Override model with custom parameters
        override_model = {
            "configuration": configuration,
            "parameters": {"max_tokens": 512}
        }

        self.prompty = Prompty.load(source=prompty_path, model=override_model)

    def __call__(self, response: str) -> dict:
        """Evaluate friendliness of response"""
        result = self.prompty(response=response)
        return {
            "friendliness": result["score"],
            "reasoning": result["reasoning"]
        }
```

#### Prompty File (friendliness.prompty)

```yaml
---
name: FriendlinessEvaluator
description: Evaluates the friendliness of a response
model:
  api: chat
  parameters:
    temperature: 0.0
    max_tokens: 512
inputs:
  response:
    type: string
outputs:
  score:
    type: float
  reasoning:
    type: string
---
system:
You are an AI assistant that evaluates the friendliness of responses.

Score the following response on a scale of 1-5 for friendliness:
- 1: Unfriendly or hostile
- 2: Neutral but cold
- 3: Polite but impersonal
- 4: Friendly and warm
- 5: Very friendly and personable

Response: {{response}}

Return your evaluation as JSON:
{
  "score": <1-5>,
  "reasoning": "<explanation>"
}
```

#### Custom Evaluator with Complex Logic

```python
class ToolCallAccuracyEvaluator:
    """Custom evaluator for tool call accuracy"""

    def __init__(self, model_config: AzureOpenAIModelConfiguration):
        self.model_config = model_config

    def __call__(
        self,
        expected_tools: list[str],
        actual_tools: list[dict],
        **kwargs
    ) -> dict:
        """Evaluate tool call accuracy"""
        # Extract tool names from actual calls
        actual_tool_names = [call['name'] for call in actual_tools]

        # Calculate metrics
        expected_set = set(expected_tools)
        actual_set = set(actual_tool_names)

        correct = len(expected_set & actual_set)
        missing = len(expected_set - actual_set)
        extra = len(actual_set - expected_set)

        total_expected = len(expected_set)

        # Calculate scores
        precision = correct / len(actual_set) if actual_set else 0
        recall = correct / total_expected if total_expected else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "tool_call_accuracy": f1,
            "precision": precision,
            "recall": recall,
            "correct": correct,
            "missing": missing,
            "extra": extra,
            "missing_tools": list(expected_set - actual_set),
            "extra_tools": list(actual_set - expected_set)
        }
```

### Retry & Error Handling

#### Built-in Retry Mechanisms

Azure services provide built-in retry logic:
- Automatic retry up to 3 times with exponential backoff
- Detection of `retry-after` headers
- Handling of transient faults

#### Custom Retry Logic

```python
import asyncio
from azure.core.exceptions import ServiceRequestError

async def evaluate_with_retry(evaluator, max_retries=3, backoff_factor=2, **kwargs):
    """Evaluate with custom retry logic"""
    for attempt in range(max_retries):
        try:
            return evaluator(**kwargs)
        except ServiceRequestError as e:
            if attempt == max_retries - 1:
                raise

            wait_time = backoff_factor ** attempt
            await asyncio.sleep(wait_time)

            print(f"Retry attempt {attempt + 1} after {wait_time}s")
        except Exception as e:
            # Non-retriable error
            raise
```

#### Error Handling Best Practices

```python
from azure.ai.evaluation import GroundednessEvaluator
from azure.core.exceptions import AzureError

def safe_evaluate(evaluator, **kwargs):
    """Evaluate with comprehensive error handling"""
    try:
        return evaluator(**kwargs)
    except ValueError as e:
        # Invalid input
        return {
            "error": "invalid_input",
            "message": str(e),
            "score": None
        }
    except AzureError as e:
        # Azure service error
        return {
            "error": "service_error",
            "message": str(e),
            "score": None
        }
    except Exception as e:
        # Unexpected error
        return {
            "error": "unexpected_error",
            "message": str(e),
            "score": None
        }
```

#### Rate Limiting

```python
import time
from collections import deque

class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old calls outside time window
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()

            # Check if we've hit the limit
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.time_window - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Make the call
            result = func(*args, **kwargs)
            self.calls.append(time.time())

            return result

        return wrapper

# Usage
@RateLimiter(max_calls=10, time_window=60)  # 10 calls per minute
def rate_limited_evaluation(evaluator, **kwargs):
    return evaluator(**kwargs)
```

### Best Practices

1. **Model Selection**
   - Use GPT-4o for critical evaluations (groundedness, relevance)
   - Use GPT-4o-mini for less critical metrics (fluency, coherence)
   - Consider cost vs. quality tradeoffs

2. **Configuration**
   - Customize grading rubrics to your scenario
   - Adjust max_tokens based on response length
   - Use query parameter in GroundednessEvaluator for better accuracy

3. **Error Handling**
   - Implement retry logic for transient failures
   - Use rate limiting to avoid throttling
   - Log all evaluation failures for debugging

4. **Batch Processing**
   - Use `evaluate()` function for batch evaluations
   - Process evaluations in parallel when possible
   - Implement progress tracking for long-running evaluations

5. **Custom Metrics**
   - Use Prompty for custom AI-assisted evaluators
   - Implement deterministic evaluators in Python
   - Document grading rubrics clearly

### Common Pitfalls

1. **Missing Credentials**: Content safety evaluators now require explicit credentials. Always pass `credential` parameter.

2. **Token Limits**: Default max_tokens (800) may be insufficient for long responses. Adjust per evaluator.

3. **Rate Limiting**: Batch evaluations can hit rate limits. Implement backoff and retry.

4. **Cost Management**: Using GPT-4o for all metrics can be expensive. Use tiered model selection.

5. **Schema Changes**: Recent SDK updates removed numpy dependency. Update code using `numpy.nan` to `math.nan`.

### Integration Considerations

For HoloDeck's test execution framework:

1. **Test Case Evaluation**
   ```python
   from azure.ai.evaluation import (
       GroundednessEvaluator,
       RelevanceEvaluator,
       AzureOpenAIModelConfiguration
   )

   async def evaluate_test_result(test_case, agent_response):
       """Evaluate agent response against test case"""
       # Configure models with tiered approach
       critical_config = AzureOpenAIModelConfiguration(
           azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
           api_key=os.getenv("AZURE_OPENAI_API_KEY"),
           azure_deployment="gpt-4o"
       )

       basic_config = AzureOpenAIModelConfiguration(
           azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
           api_key=os.getenv("AZURE_OPENAI_API_KEY"),
           azure_deployment="gpt-4o-mini"
       )

       # Create evaluators
       groundedness = GroundednessEvaluator(model_config=critical_config)
       relevance = RelevanceEvaluator(model_config=critical_config)

       # Evaluate
       results = {}

       if test_case.context:
           results["groundedness"] = safe_evaluate(
               groundedness,
               query=test_case.input,
               context=test_case.context,
               response=agent_response
           )

       results["relevance"] = safe_evaluate(
           relevance,
           query=test_case.input,
           response=agent_response
       )

       return results
   ```

2. **Batch Test Evaluation**
   ```python
   async def evaluate_test_suite(test_results):
       """Evaluate entire test suite"""
       # Prepare data for batch evaluation
       data = []
       for test in test_results:
           data.append({
               "query": test.input,
               "context": test.context,
               "response": test.output,
               "ground_truth": test.ground_truth
           })

       # Run batch evaluation
       results = evaluate(
           data=data,
           evaluators={
               "groundedness": groundedness,
               "relevance": relevance,
               "similarity": similarity
           },
           evaluator_config={
               "groundedness": {
                   "column_mapping": {
                       "query": "query",
                       "context": "context",
                       "response": "response"
                   }
               }
           }
       )

       return results
   ```

3. **Tool Call Validation**
   ```python
   async def validate_tool_calls(test_case, agent_result):
       """Validate tool calls against expected tools"""
       if not test_case.expected_tools:
           return None

       tool_evaluator = ToolCallAccuracyEvaluator(model_config=model_config)

       return tool_evaluator(
           expected_tools=test_case.expected_tools,
           actual_tools=agent_result.tool_calls
       )
   ```

4. **Threshold-Based Pass/Fail**
   ```python
   def check_evaluation_thresholds(evaluation_results, thresholds):
       """Check if evaluation results meet thresholds"""
       passed = True
       failures = []

       for metric, threshold in thresholds.items():
           if metric in evaluation_results:
               score = evaluation_results[metric]
               if score < threshold:
                   passed = False
                   failures.append({
                       "metric": metric,
                       "score": score,
                       "threshold": threshold
                   })

       return {
           "passed": passed,
           "failures": failures
       }
   ```

---

## 4. NLP Metrics Libraries

### Overview

For traditional NLP metrics (F1, BLEU, ROUGE, METEOR), there are several library options. Based on research, the **Hugging Face `evaluate` library** is the recommended choice for 2025 as it provides a unified interface, modern implementation, and better integration with LLM workflows.

### Library Comparison

| Feature | evaluate (HF) | NLTK | rouge-score | bert-score |
|---------|---------------|------|-------------|------------|
| **BLEU** | ✅ | ✅ | ❌ | ❌ |
| **ROUGE** | ✅ | ❌ | ✅ | ❌ |
| **METEOR** | ✅ | ✅ | ❌ | ❌ |
| **F1** | ✅ | ✅ | ✅ | ❌ |
| **BERTScore** | ✅ | ❌ | ❌ | ✅ |
| **Unified API** | ✅ | ❌ | ❌ | ❌ |
| **Maintained** | ✅ | ✅ | ⚠️ | ✅ |
| **Easy to Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Integration** | Excellent | Good | Basic | Good |

### Recommended: Hugging Face `evaluate`

#### Installation

```bash
# Basic installation
pip install evaluate

# With specific dependencies
pip install evaluate[template]

# For BLEU
pip install evaluate sacrebleu
```

**Requirements**:
- Python >= 3.7
- License: Apache 2.0

#### Usage Patterns

##### BLEU Score

```python
import evaluate

# Load BLEU metric
bleu = evaluate.load("bleu")

# Single prediction
predictions = ["The cat sat on the mat"]
references = [["The cat is on the mat"]]

results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU: {results['bleu']}")  # 0.0-1.0

# Multiple predictions
predictions = [
    "The cat sat on the mat",
    "I love machine learning"
]
references = [
    ["The cat is on the mat"],
    ["I enjoy machine learning"]
]

results = bleu.compute(predictions=predictions, references=references)
```

##### SacreBLEU (Recommended Variant)

```python
import evaluate

# Load SacreBLEU (more reproducible)
sacrebleu = evaluate.load("sacrebleu")

predictions = ["The cat sat on the mat"]
references = [["The cat is on the mat"]]

results = sacrebleu.compute(predictions=predictions, references=references)
print(f"SacreBLEU: {results['score']}")  # 0-100 scale
```

##### ROUGE Score

```python
import evaluate

# Load ROUGE metric
rouge = evaluate.load("rouge")

predictions = ["The cat sat on the mat"]
references = ["The cat is on the mat"]

results = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1: {results['rouge1']}")  # F1 score
print(f"ROUGE-2: {results['rouge2']}")  # F1 score
print(f"ROUGE-L: {results['rougeL']}")  # F1 score
print(f"ROUGE-Lsum: {results['rougeLsum']}")  # F1 score
```

##### METEOR Score

```python
import evaluate

# Load METEOR metric
meteor = evaluate.load("meteor")

predictions = ["The cat sat on the mat"]
references = ["The cat is on the mat"]

results = meteor.compute(predictions=predictions, references=references)
print(f"METEOR: {results['meteor']}")  # 0.0-1.0
```

##### F1 Score (for classification)

```python
import evaluate

# Load F1 metric
f1 = evaluate.load("f1")

# Binary classification
predictions = [0, 1, 0, 1, 1]
references = [0, 1, 0, 0, 1]

results = f1.compute(predictions=predictions, references=references)
print(f"F1: {results['f1']}")

# Multi-class with averaging
results = f1.compute(
    predictions=predictions,
    references=references,
    average="weighted"  # or 'macro', 'micro', 'weighted'
)
```

##### BERTScore

```python
import evaluate

# Load BERTScore
bertscore = evaluate.load("bertscore")

predictions = ["The cat sat on the mat"]
references = ["The cat is on the mat"]

results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"
)
print(f"Precision: {results['precision'][0]}")
print(f"Recall: {results['recall'][0]}")
print(f"F1: {results['f1'][0]}")
```

### Alternative: NLTK (for specific use cases)

#### Installation

```bash
pip install nltk

# Download required data
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

#### BLEU with NLTK

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Single sentence BLEU
reference = [["the", "cat", "is", "on", "the", "mat"]]
candidate = ["the", "cat", "sat", "on", "the", "mat"]

score = sentence_bleu(reference, candidate)
print(f"BLEU: {score}")

# Corpus BLEU
references = [[["the", "cat", "is", "on", "the", "mat"]]]
candidates = [["the", "cat", "sat", "on", "the", "mat"]]

score = corpus_bleu(references, candidates)
```

#### METEOR with NLTK

```python
from nltk.translate.meteor_score import meteor_score

reference = "the cat is on the mat"
candidate = "the cat sat on the mat"

score = meteor_score([reference], candidate)
print(f"METEOR: {score}")
```

### Metric Characteristics

#### BLEU (Bilingual Evaluation Understudy)

- **Primary Use**: Machine translation
- **Focus**: Precision (exact word/phrase matches)
- **Range**: 0.0-1.0 (or 0-100 for SacreBLEU)
- **Best For**: Evaluating translation quality
- **Limitations**: Doesn't capture semantic meaning

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- **Primary Use**: Text summarization
- **Focus**: Recall (capturing key information)
- **Variants**:
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence
  - ROUGE-Lsum: LCS for summaries
- **Range**: 0.0-1.0 (F1 score)
- **Best For**: Summarization tasks
- **Limitations**: Doesn't capture semantic meaning

#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

- **Primary Use**: Machine translation, paraphrase detection
- **Focus**: Harmonic mean of precision and recall (weighted toward recall)
- **Features**: Considers synonyms, stemming, paraphrasing
- **Range**: 0.0-1.0
- **Best For**: Tasks requiring nuanced language understanding
- **Advantage**: Better correlation with human judgment than BLEU

#### F1 Score

- **Primary Use**: Classification tasks
- **Focus**: Balance of precision and recall
- **Range**: 0.0-1.0
- **Best For**: Imbalanced datasets
- **Variants**: Binary, multi-class, weighted, macro, micro

#### BERTScore

- **Primary Use**: Modern text generation evaluation
- **Focus**: Semantic similarity using contextual embeddings
- **Range**: 0.0-1.0
- **Best For**: Capturing semantic meaning beyond word overlap
- **Advantage**: High correlation with human evaluation
- **Limitation**: Requires model inference (slower, more resource-intensive)

### Best Practices

1. **Library Selection**
   - Use `evaluate` for unified interface and modern implementation
   - Use NLTK only if you need specific features not in `evaluate`
   - Avoid mixing libraries for the same metric

2. **Metric Selection**
   - BLEU: Machine translation
   - ROUGE: Summarization
   - METEOR: Translation + paraphrase
   - BERTScore: Semantic similarity
   - Use multiple metrics for comprehensive evaluation

3. **Preprocessing**
   - Tokenize consistently across predictions and references
   - Lowercase for case-insensitive comparison
   - Remove punctuation if appropriate
   - Handle whitespace carefully

4. **Multiple References**
   - Provide multiple reference answers when possible
   - Improves metric reliability
   - Reduces bias from single reference

5. **Interpretation**
   - No single metric is perfect
   - Combine with human evaluation
   - Understand metric limitations
   - Set appropriate thresholds based on task

### Common Pitfalls

1. **Reference Format**: Different libraries expect different formats (list of lists vs. single list). Always check documentation.

2. **Tokenization**: Inconsistent tokenization between prediction and reference leads to incorrect scores.

3. **Case Sensitivity**: Some implementations are case-sensitive by default. Normalize text when appropriate.

4. **Over-Reliance**: BLEU/ROUGE don't capture semantic meaning. Use BERTScore or human evaluation for nuanced assessment.

5. **Single Reference Bias**: Using only one reference can penalize valid alternative answers.

### Integration Considerations

For HoloDeck's test execution framework:

1. **Metric Calculation**
   ```python
   import evaluate

   class NLPMetrics:
       """NLP metric calculator"""

       def __init__(self):
           self.bleu = evaluate.load("sacrebleu")
           self.rouge = evaluate.load("rouge")
           self.meteor = evaluate.load("meteor")
           self.bertscore = evaluate.load("bertscore")

       def calculate_all(self, prediction: str, reference: str) -> dict:
           """Calculate all NLP metrics"""
           return {
               "bleu": self.calculate_bleu(prediction, reference),
               "rouge": self.calculate_rouge(prediction, reference),
               "meteor": self.calculate_meteor(prediction, reference),
               "bertscore": self.calculate_bertscore(prediction, reference)
           }

       def calculate_bleu(self, prediction: str, reference: str) -> float:
           """Calculate BLEU score"""
           result = self.bleu.compute(
               predictions=[prediction],
               references=[[reference]]
           )
           return result['score'] / 100  # Normalize to 0-1

       def calculate_rouge(self, prediction: str, reference: str) -> dict:
           """Calculate ROUGE scores"""
           result = self.rouge.compute(
               predictions=[prediction],
               references=[reference]
           )
           return {
               "rouge1": result['rouge1'],
               "rouge2": result['rouge2'],
               "rougeL": result['rougeL']
           }

       def calculate_meteor(self, prediction: str, reference: str) -> float:
           """Calculate METEOR score"""
           result = self.meteor.compute(
               predictions=[prediction],
               references=[reference]
           )
           return result['meteor']

       def calculate_bertscore(self, prediction: str, reference: str) -> dict:
           """Calculate BERTScore"""
           result = self.bertscore.compute(
               predictions=[prediction],
               references=[reference],
               lang="en"
           )
           return {
               "precision": result['precision'][0],
               "recall": result['recall'][0],
               "f1": result['f1'][0]
           }
   ```

2. **Test Case Evaluation**
   ```python
   async def evaluate_test_with_nlp_metrics(test_case, agent_response):
       """Evaluate test case with NLP metrics"""
       if not test_case.ground_truth:
           return None

       metrics = NLPMetrics()

       # Calculate metrics
       results = metrics.calculate_all(
           prediction=agent_response,
           reference=test_case.ground_truth
       )

       return results
   ```

3. **Batch Metric Calculation**
   ```python
   async def calculate_metrics_batch(test_results):
       """Calculate metrics for batch of test results"""
       predictions = [r.output for r in test_results]
       references = [r.ground_truth for r in test_results]

       # Calculate metrics in batch (more efficient)
       bleu = evaluate.load("sacrebleu")
       rouge = evaluate.load("rouge")

       bleu_results = bleu.compute(
           predictions=predictions,
           references=[[ref] for ref in references]
       )

       rouge_results = rouge.compute(
           predictions=predictions,
           references=references
       )

       return {
           "bleu": bleu_results['score'] / 100,
           "rouge": rouge_results
       }
   ```

4. **Multiple Reference Support**
   ```python
   async def evaluate_with_multiple_references(prediction, references):
       """Evaluate against multiple reference answers"""
       bleu = evaluate.load("sacrebleu")

       # BLEU supports multiple references
       result = bleu.compute(
           predictions=[prediction],
           references=[references]  # List of multiple references
       )

       return result['score'] / 100
   ```

---

## 5. Integration Architecture Recommendations

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 CLI Layer (holodeck test)                │
│  Test Runner Entry Point                                 │
└─────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Test Execution Engine                       │
│  ├─ Test Case Loader                                     │
│  ├─ File Preprocessor (markitdown)                       │
│  ├─ Agent Executor (Semantic Kernel)                     │
│  ├─ Evaluation Engine                                    │
│  └─ Report Generator                                     │
└─────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              File Preprocessing Layer                    │
│  ├─ MarkItDown Converter                                 │
│  ├─ File Type Detector                                   │
│  ├─ Error Handler                                        │
│  └─ Cache Manager                                        │
└─────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Agent Execution Layer                       │
│  ├─ Semantic Kernel Agent                                │
│  ├─ Chat History Manager                                 │
│  ├─ Tool Call Tracker                                    │
│  └─ Timeout/Retry Handler                                │
└─────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Evaluation Layer                            │
│  ├─ Azure AI Evaluators (Groundedness, Relevance, etc.) │
│  ├─ NLP Metrics (evaluate library)                       │
│  ├─ Custom Evaluators                                    │
│  └─ Threshold Checker                                    │
└─────────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Results & Reporting                         │
│  ├─ JSON Results                                         │
│  ├─ HTML Report                                          │
│  ├─ Console Output                                       │
│  └─ Azure AI Foundry Integration                         │
└─────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. Test Execution Engine

```python
class TestExecutionEngine:
    """Main test execution orchestrator"""

    def __init__(self, config):
        self.config = config
        self.file_processor = FileProcessor()
        self.agent_executor = AgentExecutor()
        self.evaluator = EvaluationEngine()

    async def run_tests(self, test_cases: list) -> TestResults:
        """Execute all test cases"""
        results = []

        for test_case in test_cases:
            result = await self.run_single_test(test_case)
            results.append(result)

        return TestResults(results)

    async def run_single_test(self, test_case: TestCase) -> TestResult:
        """Execute single test case"""
        # 1. Preprocess files
        file_context = await self.file_processor.process_files(test_case.files)

        # 2. Execute agent
        agent_result = await self.agent_executor.execute(
            test_case=test_case,
            file_context=file_context
        )

        # 3. Evaluate results
        evaluation = await self.evaluator.evaluate(
            test_case=test_case,
            agent_result=agent_result
        )

        # 4. Return combined result
        return TestResult(
            test_case=test_case,
            agent_result=agent_result,
            evaluation=evaluation
        )
```

#### 2. File Preprocessing Layer

```python
class FileProcessor:
    """File preprocessing with markitdown"""

    def __init__(self):
        self.converter = MarkItDown()
        self.cache = {}

    async def process_files(self, file_inputs: list) -> str:
        """Process all file inputs to markdown"""
        contexts = []

        for file_input in file_inputs:
            # Check cache
            cache_key = self._get_cache_key(file_input)
            if cache_key in self.cache:
                context = self.cache[cache_key]
            else:
                # Convert file
                context = await self._convert_file(file_input)
                self.cache[cache_key] = context

            contexts.append(context)

        return "\n\n".join(contexts)

    async def _convert_file(self, file_input: FileInput) -> str:
        """Convert single file to markdown"""
        try:
            # Validate file
            validate_file_before_conversion(file_input.path)

            # Handle file type specific logic
            if file_input.type == "excel" and file_input.sheet:
                return await self._convert_excel_sheet(file_input)
            elif file_input.type == "pdf" and file_input.pages:
                return await self._convert_pdf_pages(file_input)
            else:
                # Standard conversion
                result = self.converter.convert(file_input.path)
                return result.text_content

        except Exception as e:
            return f"[ERROR: Failed to convert {file_input.path}: {e}]"
```

#### 3. Agent Execution Layer

```python
class AgentExecutor:
    """Agent execution with Semantic Kernel"""

    def __init__(self, agent_config):
        self.agent = self._create_agent(agent_config)

    async def execute(
        self,
        test_case: TestCase,
        file_context: str
    ) -> AgentResult:
        """Execute agent with test input"""
        # Create chat history
        history = ChatHistory()
        history.add_system_message(self.agent.instructions)

        # Add file context if present
        if file_context:
            history.add_user_message(f"Context:\n\n{file_context}")

        # Add test input
        history.add_user_message(test_case.input)

        # Execute with retry and timeout
        try:
            response = await self._execute_with_retry(history)

            return AgentResult(
                output=response.content,
                tool_calls=self._extract_tool_calls(response),
                status="success"
            )

        except Exception as e:
            return AgentResult(
                output=None,
                error=str(e),
                status="error"
            )

    async def _execute_with_retry(
        self,
        history: ChatHistory,
        max_retries: int = 3
    ):
        """Execute agent with retry logic"""
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(
                    self.agent.invoke(messages=history),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
```

#### 4. Evaluation Layer

```python
class EvaluationEngine:
    """Evaluation engine combining Azure AI and NLP metrics"""

    def __init__(self, config):
        self.config = config
        self.azure_evaluators = self._create_azure_evaluators()
        self.nlp_metrics = NLPMetrics()

    async def evaluate(
        self,
        test_case: TestCase,
        agent_result: AgentResult
    ) -> EvaluationResult:
        """Evaluate agent result"""
        results = {}

        # Azure AI metrics
        if test_case.evaluations:
            azure_results = await self._evaluate_azure_metrics(
                test_case,
                agent_result
            )
            results.update(azure_results)

        # NLP metrics
        if test_case.ground_truth:
            nlp_results = await self._evaluate_nlp_metrics(
                test_case.ground_truth,
                agent_result.output
            )
            results.update(nlp_results)

        # Tool call validation
        if test_case.expected_tools:
            tool_results = await self._validate_tool_calls(
                test_case.expected_tools,
                agent_result.tool_calls
            )
            results.update(tool_results)

        # Check thresholds
        passed = self._check_thresholds(results, test_case.thresholds)

        return EvaluationResult(
            metrics=results,
            passed=passed
        )

    async def _evaluate_azure_metrics(self, test_case, agent_result):
        """Evaluate using Azure AI evaluators"""
        results = {}

        for metric in test_case.evaluations.metrics:
            evaluator = self.azure_evaluators[metric.name]

            try:
                result = await evaluate_with_retry(
                    evaluator,
                    query=test_case.input,
                    context=test_case.context,
                    response=agent_result.output
                )
                results[metric.name] = result
            except Exception as e:
                results[metric.name] = {"error": str(e)}

        return results

    async def _evaluate_nlp_metrics(self, ground_truth, prediction):
        """Evaluate using NLP metrics"""
        return self.nlp_metrics.calculate_all(prediction, ground_truth)
```

### Configuration Pattern

```yaml
# test_config.yaml
agent:
  name: "research-assistant"
  config: "agent.yaml"

evaluations:
  model:
    provider: azure_openai
    azure_endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    azure_deployment: "gpt-4o-mini"  # Default for all metrics

  metrics:
    - name: groundedness
      enabled: true
      model:
        azure_deployment: "gpt-4o"  # Override for critical metric
      threshold: 0.8

    - name: relevance
      enabled: true
      model:
        azure_deployment: "gpt-4o"  # Override for critical metric
      threshold: 0.7

    - name: coherence
      enabled: true
      # Uses default model (gpt-4o-mini)
      threshold: 0.6

    - name: bleu
      enabled: true
      threshold: 0.5

    - name: rouge
      enabled: true
      threshold: 0.6

test_cases:
  - name: "Research Query with PDF"
    input: "What are the key findings?"
    files:
      - path: "data/research_paper.pdf"
        type: pdf
        pages: [1, 5]  # Pages 1-5
    expected_tools:
      - "search_papers"
      - "extract_citations"
    ground_truth: "The study found..."
    context: "Academic research paper..."
```

### Dependency Installation

```bash
# Create requirements.txt or add to pyproject.toml

# Semantic Kernel
semantic-kernel[azure]>=1.37.0

# markitdown
markitdown[all]>=0.1.0

# Azure AI Evaluation
azure-ai-evaluation>=1.0.0

# NLP Metrics
evaluate>=0.4.0
sacrebleu>=2.3.0

# Additional utilities
aiofiles>=23.0.0
httpx>=0.25.0
```

### Error Handling Strategy

```python
class TestExecutionError(Exception):
    """Base exception for test execution"""
    pass

class FileConversionError(TestExecutionError):
    """File conversion failed"""
    pass

class AgentExecutionError(TestExecutionError):
    """Agent execution failed"""
    pass

class EvaluationError(TestExecutionError):
    """Evaluation failed"""
    pass

# Comprehensive error handling
async def run_test_with_error_handling(test_case):
    """Run test with comprehensive error handling"""
    result = TestResult(test_case=test_case)

    try:
        # Execute test
        result = await execute_test(test_case)
    except FileConversionError as e:
        result.status = "file_error"
        result.error = str(e)
    except AgentExecutionError as e:
        result.status = "agent_error"
        result.error = str(e)
    except EvaluationError as e:
        result.status = "evaluation_error"
        result.error = str(e)
    except Exception as e:
        result.status = "unknown_error"
        result.error = str(e)

    return result
```

---

## 6. Decision Matrix

### Technology Selections

| Component | Decision | Rationale | Alternatives Considered |
|-----------|----------|-----------|-------------------------|
| **Agent Framework** | Semantic Kernel | - Microsoft's official SDK<br>- Native Azure integration<br>- Strong agent abstractions<br>- Multi-agent orchestration<br>- Active development | - LangChain: More complex, less Azure-native<br>- AutoGen: More research-focused<br>- Custom: Too much overhead |
| **Document Conversion** | markitdown | - Microsoft's official tool<br>- Simple API<br>- Wide format support<br>- OCR + LLM integration<br>- Growing ecosystem | - textract: More complex setup<br>- apache-tika: Java dependency<br>- docx2txt/pdfplumber: Format-specific |
| **AI Evaluation** | Azure AI Evaluation SDK | - Native Azure integration<br>- Built-in quality metrics<br>- Per-metric model override<br>- Maintained by Microsoft<br>- Designed for LLMs | - Custom prompts: Less standardized<br>- LangSmith: Separate platform<br>- TruLens: More complex |
| **NLP Metrics** | Hugging Face `evaluate` | - Unified API<br>- Modern implementation<br>- All metrics in one place<br>- Active maintenance<br>- Better than NLTK | - NLTK: Older, less unified<br>- rouge-score: Single-purpose<br>- bert-score: Single metric |

### Key Decision Points

#### 1. Why Semantic Kernel over LangChain?

**Semantic Kernel Advantages:**
- Better Azure integration (native support)
- Cleaner agent abstractions
- Built-in orchestration patterns
- Microsoft support and roadmap alignment
- Less complex than LangChain
- Better for production deployments

**When to Consider LangChain:**
- Need broader ecosystem (more community plugins)
- Using multiple LLM providers heavily
- Want more community examples

#### 2. Why markitdown over alternatives?

**markitdown Advantages:**
- Official Microsoft tool (aligned with our stack)
- Simple API (one function call)
- Wide format support (Office, PDF, images, audio)
- LLM integration for image descriptions
- Growing popularity (25k+ stars)

**When to Consider Alternatives:**
- textract: Need OCR-specific features
- apache-tika: Already using Java stack
- Format-specific libraries: Only processing one format

#### 3. Why Azure AI Evaluation over custom prompts?

**Azure AI Evaluation Advantages:**
- Standardized metrics with research backing
- Per-metric model configuration
- Maintained and updated by Microsoft
- Integration with Azure AI Foundry
- Consistent scoring rubrics

**When to Consider Custom:**
- Very domain-specific evaluation needs
- Custom grading rubrics required
- Cost optimization (can be cheaper)

#### 4. Why Hugging Face evaluate over NLTK?

**evaluate Advantages:**
- Single unified API (one library for all metrics)
- Modern implementation
- Better maintained
- SacreBLEU (more reproducible)
- BERTScore included

**When to Consider NLTK:**
- Already using NLTK extensively
- Need NLTK's linguistic features beyond metrics
- Stability over modernity

### Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| **Semantic Kernel Breaking Changes** | - Pin major version<br>- Monitor changelog<br>- Test upgrades in staging |
| **markitdown Format Support** | - Implement fallback converters<br>- Validate files before conversion<br>- Graceful error handling |
| **Azure AI Evaluation Costs** | - Use tiered model approach (GPT-4o for critical, GPT-4o-mini for others)<br>- Cache evaluation results<br>- Implement budget alerts |
| **NLP Metric Limitations** | - Use multiple metrics<br>- Combine with human evaluation<br>- Don't rely solely on scores |
| **Rate Limiting** | - Implement retry with exponential backoff<br>- Use batch APIs when available<br>- Rate limiting layer |
| **File Conversion Failures** | - Multi-tier fallback strategy<br>- Comprehensive error handling<br>- Log failures for debugging |

### Performance Considerations

| Component | Performance Impact | Optimization |
|-----------|-------------------|--------------|
| **Semantic Kernel** | Low-Medium | - Use async/await<br>- Configure reasonable timeouts<br>- Implement connection pooling |
| **markitdown** | Medium-High (large files) | - Use streaming for large files<br>- Implement caching<br>- Process files in parallel |
| **Azure AI Evaluation** | High (LLM calls) | - Batch evaluations<br>- Use cheaper models where appropriate<br>- Parallel evaluation calls |
| **NLP Metrics** | Low | - Batch calculations<br>- Use SacreBLEU (faster)<br>- Cache tokenization |

### Cost Considerations

| Component | Cost | Optimization Strategy |
|-----------|------|----------------------|
| **Semantic Kernel** | Variable (LLM calls) | - Optimize prompts<br>- Use cheaper models where appropriate<br>- Implement caching |
| **markitdown** | Free (library)<br>$$$ (LLM for images) | - Use OCR instead of LLM when possible<br>- Only use vision for critical images |
| **Azure AI Evaluation** | $$$ (LLM calls per metric) | - Tiered model approach<br>- Select metrics strategically<br>- Batch evaluations |
| **evaluate** | Free | - No optimization needed |

---

## Conclusion

### Summary of Recommendations

1. **Semantic Kernel**: Use for agent execution with native Azure integration and strong abstractions
2. **markitdown**: Use for document conversion with simple API and wide format support
3. **Azure AI Evaluation SDK**: Use for AI-assisted quality metrics with per-metric model configuration
4. **Hugging Face evaluate**: Use for NLP metrics with unified API and modern implementation

### Implementation Priority

**Phase 1: Core Execution** (Weeks 1-2)
1. Implement Semantic Kernel agent execution
2. Add basic file preprocessing with markitdown
3. Capture agent responses and tool calls

**Phase 2: Basic Evaluation** (Weeks 3-4)
4. Implement NLP metrics with `evaluate`
5. Add basic Azure AI Evaluation (groundedness, relevance)
6. Tool call validation

**Phase 3: Advanced Features** (Weeks 5-6)
7. Per-metric model configuration
8. Advanced file preprocessing (page ranges, sheet selection)
9. Caching and optimization

**Phase 4: Polish & Production** (Weeks 7-8)
10. Comprehensive error handling
11. Rate limiting and retry logic
12. Performance optimization
13. Documentation and examples

### Next Steps

1. **Set up development environment**
   ```bash
   pip install semantic-kernel[azure] markitdown[all] azure-ai-evaluation evaluate sacrebleu
   ```

2. **Create proof of concept**
   - Simple test case execution
   - Basic evaluation
   - Error handling

3. **Iterate on architecture**
   - Refine based on POC learnings
   - Add missing features
   - Optimize performance

4. **Production hardening**
   - Comprehensive testing
   - Error recovery
   - Monitoring and logging

### References

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [markitdown GitHub](https://github.com/microsoft/markitdown)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
- [Hugging Face evaluate](https://huggingface.co/docs/evaluate/)
- [GenAI Evaluation Metrics Guide](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-metrics-built-in)

---

**Document Version**: 1.0
**Last Updated**: November 1, 2025
**Author**: Research compiled for HoloDeck project
