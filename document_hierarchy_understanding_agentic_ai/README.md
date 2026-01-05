# 🤖 Hierarchical Document RAG with AutoGen

A production-ready multi-agent RAG system that uses specialized AI agents to collaboratively answer questions about hierarchical documents.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![AutoGen](https://img.shields.io/badge/Framework-AutoGen-purple)
![Anthropic](https://img.shields.io/badge/Powered%20by-Claude%20API-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Why Multi-Agent?

Traditional RAG uses a single pipeline: embed → retrieve → generate. This works for simple queries but struggles with:

- **Complex queries** requiring multiple search strategies
- **Cross-references** between documents  
- **Verification** of retrieved information
- **Context-aware** answer generation

Our multi-agent approach assigns specialized roles:

| Agent | Role |
|-------|------|
| 🎯 **Coordinator** | Understands queries, delegates tasks, synthesizes final answers |
| 🔍 **Retriever** | Searches with hierarchy awareness, expands context |
| 🔗 **CrossReference** | Detects and follows document references |
| 📝 **Summarizer** | Synthesizes information from multiple sources |
| ✅ **Verifier** | Checks accuracy, ensures proper citations |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                  │
│                "What are the termination conditions?"               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   🎯 COORDINATOR AGENT                              │
│  "This is a specific query about legal terms. I'll have            │
│   Retriever search for termination clauses."                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   🔍 RETRIEVER AGENT                                │
│  search_content("termination conditions")                          │
│  get_parent_context(matched_node)                                  │
│  "Found Section 5: Termination with 3 relevant paragraphs"         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   🔗 CROSSREF AGENT                                 │
│  find_references_in_node(section_5)                                │
│  "Found reference to Section 3.2 (Payment Terms)"                  │
│  get_related_nodes(section_5)                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   📝 SUMMARIZER AGENT                               │
│  "Based on Section 5 and related Section 3.2:                      │
│   1. Standard: 30 days notice                                      │
│   2. Immediate: material breach, bankruptcy, non-payment"          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ✅ VERIFIER AGENT                                 │
│  "✓ Verified: 30-day notice from Section 5.1                       │
│   ✓ Verified: Immediate termination from Section 5.2               │
│   ✓ Cross-reference to 3.2 correctly cited"                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FINAL ANSWER                                  │
│  "According to Section 5 (Termination), there are two types..."    │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/hierarchical-rag-autogen.git
cd hierarchical-rag-autogen

pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-api-key"
```

### Basic Usage

```python
from hierarchical_rag_autogen import HierarchicalRAGAutoGen, RAGConfig

# Simple two-agent setup (fastest)
rag = HierarchicalRAGAutoGen()
rag.add_document("contract.md")
rag.build_index()

answer = rag.query("What are the payment terms?")
print(answer)
```

### Full Multi-Agent Setup

```python
from hierarchical_rag_autogen import HierarchicalRAGAutoGen, RAGConfig

# Full group chat with all agents
config = RAGConfig(
    workflow_type="group_chat",  # Full multi-agent collaboration
    model="claude-sonnet-4-20250514",
    verbose=True
)

rag = HierarchicalRAGAutoGen(config)
rag.add_documents_from_directory("./legal_docs")
rag.build_index()

# Get answer with full agent conversation
result = rag.query_with_details(
    "Compare the liability clauses across all contracts"
)

print(result["answer"])

# See how agents collaborated
for msg in result["conversation"]:
    print(f"[{msg['name']}]: {msg['content'][:200]}...")
```

### Command Line

```bash
# Simple query
python main.py --file contract.md query "What is the summary?"

# Full multi-agent with group chat
python main.py --workflow group_chat --directory ./docs interactive

# Sequential workflow (more predictable)
python main.py --workflow sequential --file report.md query "Key findings?"

# Demo mode
python main.py demo
```

## 📁 Project Structure

```
hierarchical-rag-autogen/
├── hierarchical_rag_autogen/
│   ├── __init__.py          # Package exports
│   ├── rag.py                # Main HierarchicalRAGAutoGen class
│   ├── agents.py             # Agent definitions and prompts
│   ├── orchestrator.py       # Group chat and workflow orchestration
│   ├── tools.py              # Tool functions for agents
│   └── core/
│       ├── models.py         # Data classes
│       ├── parser.py         # Document parsing
│       └── vector_store.py   # ChromaDB integration
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## 🔧 Workflow Types

### 1. Group Chat (Most Capable)

```python
config = RAGConfig(workflow_type="group_chat")
```

- Agents discuss in a group chat
- Dynamic speaker selection based on conversation
- Best for complex, multi-step queries
- Higher token usage

### 2. Sequential (Predictable)

```python
config = RAGConfig(workflow_type="sequential")
```

- Fixed sequence: Coordinator → Retriever → Summarizer → Verifier
- Easier to debug and understand
- Good balance of capability and efficiency

### 3. Simple (Fastest)

```python
config = RAGConfig(workflow_type="simple")
```

- Two agents: Assistant + User Proxy
- Single agent does retrieval and answering
- Lowest latency and token usage
- Best for straightforward queries

## 🛠️ Agent Tools

Agents have access to these tools:

### Document Management
- `add_document(file_path)` - Add a document
- `list_documents()` - List all documents
- `get_document_structure(doc_id)` - Get section tree

### Retrieval
- `search_content(query, n_results, node_type)` - Vector search
- `search_summaries(query)` - Search document/section summaries
- `get_node_content(node_id)` - Get full node content
- `get_parent_context(node_id)` - Get parent chain
- `get_siblings(node_id)` - Get adjacent nodes

### Cross-References
- `find_references_in_node(node_id)` - Detect references
- `get_related_nodes(node_id, depth)` - Follow reference chains

### Indexing
- `build_index()` - Build vector index
- `get_index_stats()` - Get index statistics

## 📖 Customizing Agents

### Modify Agent Prompts

```python
from hierarchical_rag_autogen.agents import create_retriever_agent

# Custom retriever with domain-specific instructions
CUSTOM_PROMPT = """You are a legal document specialist...
Focus on: contract clauses, defined terms, cross-references..."""

agent = create_retriever_agent(
    llm_config={"model": "claude-sonnet-4-20250514"},
    tool_functions=tool_functions
)
agent.update_system_message(CUSTOM_PROMPT)
```

### Add Custom Tools

```python
from hierarchical_rag_autogen.tools import get_parser

def search_defined_terms(term: str) -> str:
    """Search for defined terms in the document."""
    parser = get_parser()
    # Custom search logic...
    return results

# Register with agents
tool_functions["search_defined_terms"] = search_defined_terms
```

### Create Custom Agents

```python
from autogen import AssistantAgent

LEGAL_EXPERT_PROMPT = """You are a legal expert agent..."""

legal_expert = AssistantAgent(
    name="LegalExpert",
    system_message=LEGAL_EXPERT_PROMPT,
    llm_config=llm_config,
)
```

## 🔬 Example: Complex Query Flow

```
User: "How do the termination and liability clauses interact?"

[Coordinator]: This query requires understanding two separate sections
and their relationship. I'll have Retriever find both sections.

[Retriever]: 
  - search_content("termination clauses") → Section 5
  - search_content("liability clauses") → Section 7
  - get_parent_context for both sections
  
Found relevant sections with their context.

[CrossReference]:
  - find_references_in_node(section_5) → References Section 7.2
  - find_references_in_node(section_7) → References Section 5.3
  
These sections are cross-linked.

[Summarizer]: Based on retrieved content:
- Section 5.3: "Termination does not limit liability claims"
- Section 7.2: "Liability cap applies even after termination"
These clauses are designed to work together...

[Verifier]: 
  ✓ Section 5.3 quote verified
  ✓ Section 7.2 quote verified
  ✓ Cross-references correctly identified
  
All claims supported by source documents.

[Coordinator]: FINAL ANSWER:
The termination and liability clauses are interconnected:
1. Per Section 5.3, terminating the agreement doesn't waive...
2. Section 7.2 clarifies that liability caps remain...
...
```

## 📊 Comparison with Single-Agent RAG

| Aspect | Single-Agent | Multi-Agent (AutoGen) |
|--------|--------------|----------------------|
| Query Understanding | Basic | Deep (Coordinator analyzes) |
| Retrieval Strategy | Fixed | Adaptive (based on query type) |
| Cross-References | Limited | Explicit (CrossRef agent) |
| Answer Verification | None | Built-in (Verifier agent) |
| Debugging | Opaque | Transparent (conversation log) |
| Token Usage | Lower | Higher |
| Latency | Lower | Higher |

## 🛡️ Limitations

- Higher latency due to multi-agent conversation
- More token usage (each agent generates responses)
- Requires careful prompt engineering for agent coordination
- AutoGen framework learning curve

## 📄 License

MIT License - feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Custom agent types for specific domains
- Improved speaker selection algorithms
- Caching strategies for agent responses
- Streaming support


