# 💻 Code Repository RAG

Intelligent Q&A over codebases with AST-aware analysis, dependency graphs, and multi-file code tracing.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![React](https://img.shields.io/badge/React-18-blue)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌳 **AST-Aware Chunking** | Parses functions, classes, methods - not arbitrary text splits |
| 🔗 **Dependency Graph** | Tracks imports, call relationships, inheritance |
| 🧪 **Test Association** | Links code to related tests automatically |
| 🔄 **Code Flow Tracing** | "How does execution flow from A to B?" |
| 🔍 **Hybrid Search** | Semantic + symbol + text search |
| 🌐 **Web UI** | React frontend with dependency visualization |
| 🔌 **Multi-Provider** | Claude, Gemini, GPT + Voyage, OpenAI embeddings |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/code-rag.git
cd code-rag

pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key"  # or GOOGLE_API_KEY, OPENAI_API_KEY
```

### Start Web UI

```bash
python main.py serve
# Open http://localhost:8000
```

### Python Usage

```python
from code_rag import CodeRAG, CodeRAGConfig

# Initialize
config = CodeRAGConfig(
    llm_provider="anthropic",
    embedding_provider="local",  # or "voyage" for production
)
rag = CodeRAG(config)

# Index a codebase
rag.index_directory("./my_project")

# Query the code
response = rag.query("How does the authentication flow work?")
print(response.answer)

# See what was found
for result in response.context.primary_results:
    print(f"{result.unit.name} in {result.unit.file_path}")

# Trace execution flow
trace = rag.trace_flow("login", "create_session")
print(trace.answer)
for step in trace.code_flow:
    print(f"  {step['step']}. {step['name']}")

# Find a symbol
results = rag.find_symbol("UserAuth")

# Get callers of a function
callers = rag.get_callers("validate_token")
```

### CLI Usage

```bash
# Interactive mode
python main.py interactive ./my_project

# Single query
python main.py query "How does caching work?" ./my_project

# Trace code flow
python main.py trace request_handler send_response ./my_project

# Find symbol
python main.py find DatabaseConnection ./my_project
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SOURCE CODE                            │
│                 Python, JavaScript, TypeScript              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AST PARSER                               │
│                                                             │
│  • Functions with signatures, docstrings, parameters        │
│  • Classes with methods and inheritance                     │
│  • Imports and module dependencies                          │
│  • Call relationships                                       │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  CODE INDEX     │ │ DEPENDENCY      │ │ SYMBOL TABLE    │
│                 │ │ GRAPH           │ │                 │
│ Embeddings for  │ │ imports →       │ │ Name → Units    │
│ semantic search │ │ calls →         │ │ lookup          │
│                 │ │ extends →       │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID SEARCH                            │
│                                                             │
│  Semantic (embeddings) + Symbol (exact) + Text (grep)       │
│  + Graph traversal for related code                         │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                           │
│                                                             │
│  Code + docstrings + related tests + call chains            │
│  → Comprehensive answer with citations                      │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
code-rag/
├── code_rag/
│   ├── __init__.py
│   ├── rag.py                 # Main CodeRAG orchestrator
│   ├── providers/
│   │   ├── llm.py            # Claude, Gemini, OpenAI
│   │   └── embeddings.py     # Voyage, OpenAI, Google, local
│   ├── parsing/
│   │   ├── python_parser.py  # AST parser for Python
│   │   └── js_parser.py      # Parser for JS/TS
│   ├── graph/
│   │   ├── dependency_graph.py  # Graph data structure
│   │   └── builder.py        # Builds graph from parsed code
│   └── indexing/
│       └── code_index.py     # Vector + symbol index
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React UI
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 🔧 Configuration

```python
from code_rag import CodeRAG, CodeRAGConfig

config = CodeRAGConfig(
    # LLM
    llm_provider="anthropic",    # "anthropic", "gemini", "openai"
    llm_model=None,              # Uses default if None
    
    # Embeddings
    embedding_provider="local",   # "voyage", "openai", "google", "local"
    embedding_model=None,         # "voyage-code-2" recommended for production
    
    # Search
    n_results=10,
    include_tests=True,           # Include test code in context
    include_related=True,         # Include related code via graph
    max_related_depth=2,
    
    # Generation
    max_tokens=2000,
    temperature=0.1,
    
    # Filtering
    exclude_patterns=[
        "node_modules", "__pycache__", ".git",
        "dist", "build", "*.min.js",
    ],
)

rag = CodeRAG(config)
```

## 🌳 AST-Aware Parsing

Unlike naive text chunking, Code RAG understands code structure:

### What Gets Extracted

| Unit Type | Information Extracted |
|-----------|----------------------|
| **Functions** | Name, signature, parameters, return type, docstring, decorators, calls made |
| **Classes** | Name, base classes, docstring, decorators, methods |
| **Methods** | Same as functions + parent class |
| **Imports** | Module, names imported, aliases |

### Supported Languages

| Language | Extensions | Parser |
|----------|------------|--------|
| Python | `.py` | Full AST parsing |
| JavaScript | `.js`, `.jsx`, `.mjs` | Regex-based |
| TypeScript | `.ts`, `.tsx` | Regex-based |

## 🔗 Dependency Graph

The dependency graph tracks relationships:

```python
# Get what a function calls
callees = rag.get_callees("process_request")

# Get what calls a function
callers = rag.get_callers("validate_token")

# Find related tests
tests = rag.get_tests_for("UserAuth")

# Get imports/importers
imports = rag.graph.get_imports("auth.py")
importers = rag.graph.get_importers("utils.py")

# Trace call path
path = rag.graph.find_call_path("handle_login", "create_session")
```

### Relationship Types

- `IMPORTS` - File imports another file
- `CALLS` - Function/method calls another
- `EXTENDS` - Class extends another
- `CONTAINS` - File contains code unit
- `TESTS` - Test tests a code unit

## 🔄 Code Flow Tracing

Trace execution paths through your codebase:

```python
response = rag.trace_flow("login_handler", "send_email")

print(response.answer)  # Detailed explanation

# Step-by-step flow
for step in response.code_flow:
    print(f"{step['step']}. {step['name']} ({step['type']})")
    print(f"   File: {step['file']}")
    print(f"   Signature: {step['signature']}")
```

## 🔍 Search Methods

### Semantic Search
```python
# Find code by meaning
results = rag.index.search_semantic("handle user authentication")
```

### Symbol Search
```python
# Find by name (exact or fuzzy)
results = rag.index.search_symbol("UserAuth", exact=False)
```

### Hybrid Search
```python
# Combines semantic + symbol + text
results = rag.index.search_hybrid("validate JWT token")
```

## 📊 Example Queries

| Query | What It Does |
|-------|--------------|
| "How does authentication work?" | Finds auth-related code, traces flow |
| "What calls the database?" | Uses dependency graph to find DB callers |
| "Explain the UserService class" | Retrieves class + methods + docstrings |
| "How is data validated?" | Semantic search for validation logic |
| "What tests cover the API endpoints?" | Links tests to tested code |

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/init` | Initialize system |
| POST | `/api/index/directory` | Index a codebase |
| POST | `/api/query` | Query the code |
| POST | `/api/trace` | Trace code flow |
| POST | `/api/symbol/find` | Find a symbol |
| POST | `/api/symbol/callers` | Get callers |
| GET | `/api/graph` | Get dependency graph |

### Example API Calls

```bash
# Initialize
curl -X POST http://localhost:8000/api/init \
  -H "Content-Type: application/json" \
  -d '{"llm_provider": "anthropic"}'

# Index codebase
curl -X POST "http://localhost:8000/api/index/directory?path=/path/to/code"

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does caching work?", "k": 10}'

# Trace flow
curl -X POST http://localhost:8000/api/trace \
  -H "Content-Type: application/json" \
  -d '{"start": "handle_request", "end": "send_response"}'
```

## 🎨 Web UI Features

- **Configuration Panel**: Set providers, index directories
- **Query Interface**: Natural language questions about code
- **Flow Tracing**: Visualize execution paths
- **Symbol Search**: Find functions, classes, methods
- **Dependency Graph**: Interactive vis.js visualization

## 💡 Best Practices

1. **Use voyage-code-2 embeddings** for production (optimized for code)
2. **Index focused directories** - exclude node_modules, __pycache__, etc.
3. **Include tests** - they help explain expected behavior
4. **Use trace_flow** for "how does X connect to Y" questions
5. **Check callers/callees** for impact analysis

## 📈 Performance Tips

- **Smaller codebases** (< 1000 files) work best
- **Exclude generated code** and dependencies
- **Use local embeddings** for testing, API embeddings for production
- **Limit search results** (k=10-20) for faster responses


