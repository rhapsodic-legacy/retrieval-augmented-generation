# 🤖 Agentic RAG with Tool Use

A smart assistant that **decides when to retrieve vs. use tools vs. answer directly**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Multi-LLM](https://img.shields.io/badge/LLM-Claude%20|%20Gemini%20|%20GPT-green)
![Tools](https://img.shields.io/badge/Tools-Calculator%20|%20Web%20|%20Code-orange)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔀 **Intelligent Routing** | Classifies queries → direct answer, retrieval, tool use, or web search |
| 🔄 **Self-Reflective Retrieval** | Checks if retrieved docs actually answer the question |
| 🌐 **Web Search Fallback** | Automatically searches web when knowledge base is insufficient |
| 📝 **Citations with Confidence** | Every claim cited with source and confidence score |
| 🔧 **Built-in Tools** | Calculator, code executor, datetime, web search |
| 🎯 **Multi-LLM Support** | Claude, Gemini, OpenAI - switch with one line |
| 💻 **Multiple Interfaces** | Terminal CLI, Python API, Web UI |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key for your provider:
export GOOGLE_API_KEY="your-key"     # For Gemini (recommended - free tier)
export ANTHROPIC_API_KEY="your-key"  # For Claude
export OPENAI_API_KEY="your-key"     # For OpenAI
```

### Terminal Interface

```bash
# Interactive mode with rich output
python main.py interactive

# Single query
python main.py query "What is 125 * 48?"

# Add documents and query
python main.py interactive docs/*.txt
```

### Web Interface

```bash
python main.py serve
# Open http://localhost:8000
```

### Python API

```python
from agentic_rag import AgenticRAG, AgenticRAGConfig

# Initialize with your preferred provider
config = AgenticRAGConfig(
    llm_provider="gemini",  # or "anthropic", "openai"
)
rag = AgenticRAG(config)

# Add documents to knowledge base
rag.add_document("Our refund policy allows...", source="policy.pdf")
rag.add_file("handbook.txt")

# Query with automatic routing
response = rag.query("What is 25 * 4?")
# → Routed to calculator tool

response = rag.query("What's our refund policy?")
# → Routed to knowledge base retrieval

response = rag.query("What's the latest news on AI?")
# → Routed to web search

# Check results
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.0%}")
print(f"Route: {response.route_decision.route_type.value}")

for citation in response.citations:
    print(f"  Source: {citation.source_name} ({citation.confidence:.0%})")
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      QUERY ROUTER                                 │
│                                                                   │
│   Pattern Matching ──────────► Quick classification              │
│          │                                                        │
│          ▼                                                        │
│   LLM Classification ────────► Complex queries                   │
│                                                                   │
│   Output: RouteDecision(type, confidence, suggested_tools)       │
└──────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │    DIRECT     │   │  RETRIEVAL    │   │   TOOL_USE    │
    │               │   │               │   │               │
    │ Answer from   │   │ Search KB +   │   │ Calculator    │
    │ LLM knowledge │   │ Self-reflect  │   │ Code Exec     │
    │               │   │               │   │ Web Search    │
    └───────────────┘   └───────────────┘   └───────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                  RESPONSE GENERATOR                               │
│                                                                   │
│   Synthesize ─────► Combine sources                              │
│   Cite ───────────► Add [1], [2] citations                       │
│   Confidence ─────► Assess reliability                           │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│   AgenticResponse                                                 │
│   ├── answer: str                                                │
│   ├── confidence: float (0-1)                                    │
│   ├── citations: [Citation(source, confidence, snippet)]         │
│   ├── route_decision: RouteDecision                              │
│   ├── steps: [AgentStep(type, description, duration)]            │
│   └── tool_results: [(tool_name, ToolResult)]                    │
└──────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
agentic-rag/
├── agentic_rag/
│   ├── __init__.py
│   ├── rag.py                  # Main AgenticRAG orchestrator
│   ├── providers/
│   │   ├── llm.py             # Claude, Gemini, OpenAI with tool calling
│   │   └── embeddings.py      # Voyage, OpenAI, local embeddings
│   ├── routing/
│   │   └── router.py          # Query classification & routing
│   ├── retrieval/
│   │   └── retriever.py       # Self-reflective retrieval
│   ├── tools/
│   │   └── tool_system.py     # Tool registry & built-in tools
│   └── generation/
│       └── generator.py       # Response generation with citations
├── api.py                      # FastAPI backend
├── frontend/
│   └── index.html             # React web UI
├── main.py                     # CLI
├── requirements.txt
└── README.md
```

## 🔀 Query Routing

The router classifies queries into five categories:

| Route | Description | Example |
|-------|-------------|---------|
| **DIRECT** | Answer from LLM knowledge | "What is photosynthesis?" |
| **RETRIEVAL** | Search knowledge base | "What's our vacation policy?" |
| **TOOL_USE** | Use a specific tool | "Calculate 15% of 250" |
| **WEB_SEARCH** | Search the internet | "Latest news on AI" |
| **HYBRID** | Combine approaches | "Compare our policy with industry standards" |

### Pattern-Based Fast Path

```python
# These patterns trigger immediate routing (no LLM call needed)
CALCULATION: r'\b\d+\s*[\+\-\*\/]\s*\d+'  → calculator
DATETIME: r'\bwhat time\b|\bwhat date\b'  → datetime tool
CURRENT_INFO: r'\bcurrent\b|\btoday\b'    → web search
```

### LLM Classification

For complex queries, the router uses the LLM to classify with confidence scores.

## 🔄 Self-Reflective Retrieval

After retrieving documents, the system evaluates:

1. **Relevance**: Are docs topically related?
2. **Coverage**: Do docs contain needed information?
3. **Completeness**: Any gaps in the information?
4. **Quality**: EXCELLENT → GOOD → PARTIAL → INSUFFICIENT → NONE

If quality is insufficient, it automatically:
- Tries alternative search queries
- Falls back to web search

```python
# Example retrieval result
RetrievalResult(
    documents=[...],
    quality=RetrievalQuality.GOOD,
    coverage_score=0.85,
    gaps=["Missing recent updates"],
    needs_web_search=False,
)
```

## 🔧 Built-in Tools

| Tool | Description | Example Input |
|------|-------------|---------------|
| **calculator** | Math operations | "sqrt(16) + 2^3" |
| **datetime** | Current date/time | operation="now" |
| **web_search** | Search the web | query="AI news 2024" |
| **code_executor** | Run Python code | code="print(sum(range(10)))" |

### Adding Custom Tools

```python
from agentic_rag import create_custom_tool, ToolCategory

def my_api_call(endpoint: str, params: dict) -> dict:
    # Your API logic
    return {"result": "..."}

tool = create_custom_tool(
    name="my_api",
    description="Call my custom API",
    func=my_api_call,
    parameters={
        "type": "object",
        "properties": {
            "endpoint": {"type": "string"},
            "params": {"type": "object"}
        },
        "required": ["endpoint"]
    },
    category=ToolCategory.UTILITY,
)

rag.register_tool(tool)
```

## 📝 Citations & Confidence

Every response includes citations with confidence scores:

```python
response = rag.query("What's our refund policy?")

print(response.answer)
# "According to [1], customers can request a refund within 30 days..."

print(response.confidence)  # 0.85

for cite in response.citations:
    print(f"[{cite.source_id}] {cite.source_name}")
    print(f"    Type: {cite.source_type.value}")
    print(f"    Confidence: {cite.confidence:.0%}")
    print(f"    Snippet: {cite.content_snippet[:100]}...")
```

### Confidence Calculation

- **Source quality**: Authoritative sources score higher
- **Source agreement**: Multiple agreeing sources boost confidence
- **Completeness**: Partial answers reduce confidence
- **Recency**: Outdated information reduces confidence

## 🔧 Configuration

```python
from agentic_rag import AgenticRAG, AgenticRAGConfig

config = AgenticRAGConfig(
    # LLM Provider
    llm_provider="gemini",      # "gemini", "anthropic", "openai"
    llm_model=None,             # Uses provider default
    
    # Embeddings
    embedding_provider="local", # "local", "voyage", "openai"
    
    # Retrieval
    retrieval_k=5,              # Number of docs to retrieve
    enable_reflection=True,     # Self-reflective retrieval
    enable_relevance_assessment=True,
    
    # Tools
    enable_tools=True,
    enable_web_search=True,
    enable_code_execution=True,
    
    # Generation
    max_tokens=2000,
    temperature=0.1,
    assess_confidence=True,
)
```

## 💻 Interfaces

### CLI (Terminal)

```bash
# Interactive mode with pretty output
python main.py interactive -v

# Commands in interactive mode:
#   /add FILE    - Add document to KB
#   /tools       - List available tools
#   /stats       - Show statistics
#   /verbose     - Toggle verbose mode
#   /quit        - Exit

# Single query
python main.py query "What is 2+2?" -v

# With documents
python main.py interactive docs/*.txt
```

### Web UI

Features:
- 🎨 Dark theme with animated borders
- 📊 Confidence bar visualization
- 🔀 Route badges (color-coded)
- 📋 Step-by-step reasoning display
- 🔧 Tool results display
- 📝 Citations panel

### Python API

```python
# Basic usage
response = rag.query("Your question")

# Streaming
for chunk in rag.query_stream("Your question"):
    print(chunk, end="", flush=True)

# With step callbacks
def on_step(step):
    print(f"{step.step_type}: {step.description}")

rag.on_step = on_step
response = rag.query("Your question")
```

## 🎯 Example Queries

| Query | Route | Tools Used |
|-------|-------|------------|
| "What is 125 * 48?" | TOOL_USE | calculator |
| "What's the current time?" | TOOL_USE | datetime |
| "Search for recent AI news" | WEB_SEARCH | web_search |
| "What's our refund policy?" | RETRIEVAL | knowledge_base |
| "Explain quantum computing" | DIRECT | none |
| "Calculate revenue growth and compare to docs" | HYBRID | calculator + retrieval |

## 📈 Performance Tips

1. **Use pattern routing** - Add patterns for common query types
2. **Tune retrieval_k** - More docs = better coverage but slower
3. **Disable unused tools** - Reduces routing complexity
4. **Use local embeddings** - Faster for testing
5. **Batch add documents** - Use `add_documents()` for multiple docs

## 🔒 Safety

- Code execution is sandboxed with limited builtins
- Web search uses DuckDuckGo (no tracking)
- No data sent to third parties except LLM providers

## 📝 License

MIT License
