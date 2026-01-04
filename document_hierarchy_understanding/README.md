# 🏛️ Hierarchical Document RAG

A production-grade Retrieval-Augmented Generation system that understands document structure. Built for large document collections with complex hierarchies (legal, technical, medical, academic).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Anthropic](https://img.shields.io/badge/Powered%20by-Claude%20API-orange)
![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-green)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Key Features

### 📊 Document Hierarchy Understanding
```
Document
├── Section 1
│   ├── Subsection 1.1
│   │   ├── Paragraph
│   │   └── Paragraph
│   └── Subsection 1.2
│       └── Paragraph
└── Section 2
    └── Paragraph
```

### 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Hierarchical Parsing** | Automatically extracts Document → Section → Paragraph structure from Markdown, numbered sections, or plain text |
| **Parent-Context Retrieval** | When retrieving a paragraph, includes section and document context for better answers |
| **Cross-Document Linking** | Detects and resolves references like "See Section 3.2" or citations "[1]" |
| **Summarization Layers** | Generates summaries at each level; queries summaries first, then drills down |
| **Adaptive Retrieval** | Automatically chooses strategy based on query type (broad vs. specific) |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hierarchical-rag.git
cd hierarchical-rag

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-api-key"
```

### Basic Usage

```python
from hierarchical_rag import HierarchicalRAG

# Initialize
rag = HierarchicalRAG()

# Add documents
rag.add_document("legal_contract.md")
rag.add_document("technical_spec.md")
rag.add_documents_from_directory("./docs")

# Build index (generates embeddings + summaries)
rag.build_index()

# Query
result = rag.query("What are the termination conditions in section 5?")
print(result.answer)

# Query with drill-down (broad search → specific)
result = rag.query_with_drill_down(
    question="What is the liability cap?",
    broad_query="liability and indemnification"
)
```

### Command Line

```bash
# Index documents
python main.py index --directory ./docs --persist-dir ./index

# Query
python main.py query "What are the key findings?" --persist-dir ./index

# Interactive mode
python main.py interactive --file report.md

# Show document structure
python main.py tree --file document.md
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE RETRIEVER                               │
│  Classifies query → chooses strategy (broad/specific/cross-ref)     │
└─────────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  SUMMARY LAYER   │ │  CONTENT LAYER   │ │  CROSS-REF LAYER │
│  Query doc/sect  │ │  Query paragraphs│ │  Expand via refs │
│  summaries first │ │  with filtering  │ │  and citations   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT EXPANSION                                │
│  Retrieved paragraph + parent section + document context            │
│  + sibling paragraphs + cross-referenced nodes                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                                   │
│  Full hierarchical context → Claude → Cited answer                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
hierarchical-rag/
├── hierarchical_rag/
│   ├── __init__.py          # Package exports
│   ├── models.py            # Data classes (Document, Section, Paragraph, etc.)
│   ├── parser.py            # Document parsing into hierarchy
│   ├── vector_store.py      # ChromaDB with hierarchy metadata
│   ├── cross_references.py  # Reference detection and linking
│   ├── summarizer.py        # Multi-level summarization
│   ├── retriever.py         # Hierarchy-aware retrieval
│   └── rag.py               # Main orchestrator
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

## 📖 Module Documentation

### Document Parser

```python
from hierarchical_rag import DocumentParser

parser = DocumentParser(min_paragraph_length=50)

# Parse a file
doc = parser.parse_file("document.md")

# Parse text
doc = parser.parse_text("""
# Introduction
This is the intro paragraph.

## Background
### Historical Context
Some historical information here.
""", title="My Document")

# Access hierarchy
print(f"Sections: {doc.total_sections}")
print(f"Paragraphs: {doc.total_paragraphs}")

# Get all nodes
nodes = parser.get_all_nodes()
for node_id, node in nodes.items():
    print(f"{node.node_type}: {node.title}")
```

### Vector Store

```python
from hierarchical_rag import HierarchicalVectorStore, NodeType

store = HierarchicalVectorStore(persist_directory="./index")

# Query with filters
results = store.query(
    query_embedding=embedding,
    n_results=5,
    node_types=[NodeType.PARAGRAPH],
    document_ids=["doc_123"],
    min_level=2  # Only subsections and below
)

# Query summaries for broad search
summary_results = store.query_summaries(
    query_embedding=embedding,
    n_results=3,
    node_types=[NodeType.DOCUMENT, NodeType.SECTION]
)
```

### Cross-Reference Detection

```python
from hierarchical_rag import CrossReferenceDetector

detector = CrossReferenceDetector(all_nodes)

# Process all references
references = detector.process_all_references()

# Find related nodes
related = detector.get_related_nodes("node_123", max_depth=2)
```

### Retrieval Strategies

```python
from hierarchical_rag import HierarchicalRetriever, RetrievalConfig

config = RetrievalConfig(
    n_summary_results=3,
    n_content_results=5,
    include_parent_context=True,
    include_sibling_context=True,
    include_references=True,
    use_summary_layer=True
)

retriever = HierarchicalRetriever(vector_store, all_nodes, cross_ref_detector, config)

# Standard retrieval
results = retriever.retrieve(query_embedding)

# Summary drill-down
results = retriever.retrieve_with_summary_drill_down(
    query_embedding,
    n_summaries=3,
    n_per_summary=2
)

# Cross-reference expansion
results = retriever.retrieve_with_cross_references(
    query_embedding,
    n_results=5,
    reference_depth=2
)
```

## 🔧 Configuration

### HierarchicalRAG Options

```python
rag = HierarchicalRAG(
    persist_directory="./index",     # Persist vectors (None for in-memory)
    model="claude-sonnet-4-20250514",  # Claude model for generation
    embedding_model="voyage-3",      # Embedding model
    auto_summarize=True,             # Generate summaries on build
    verbose=True                     # Print progress
)
```

### RetrievalConfig Options

```python
config = RetrievalConfig(
    # Result counts
    n_summary_results=3,         # Summaries to check first
    n_content_results=5,         # Content results to return
    
    # Context expansion
    include_parent_context=True,  # Include section/document context
    include_sibling_context=False,# Include adjacent paragraphs
    max_sibling_count=2,          # Max siblings to include
    
    # Cross-references
    include_references=True,      # Follow cross-references
    max_reference_depth=1,        # Hops to follow
    max_references=3,             # Max referenced nodes
    
    # Summary layer
    use_summary_layer=True,       # Query summaries first
    summary_threshold=0.5,        # Min score for summary match
    
    # Filtering
    target_node_types=[NodeType.PARAGRAPH],
    target_document_ids=["doc_123"]
)
```

## 📊 Example Output

```
🔍 Query: What are the termination conditions?

══════════════════════════════════════════════════════════════

📄 Document: Service Agreement
   Summary: A comprehensive service agreement covering terms,
   liability, and termination conditions.

  📑 Section: 5. Termination
     Summary: Outlines conditions under which either party
     may terminate the agreement.

==================================================
📝 Matched Content (relevance: 0.89):
Either party may terminate this Agreement with 30 days written
notice. Immediate termination is permitted in cases of material
breach, bankruptcy, or failure to pay fees within 60 days.
==================================================

🔗 Related References:
  - [section] 3.2 Payment Terms: Payment is due within 30 days...
  - [section] 7.1 Dispute Resolution: Any disputes arising...

──────────────────────────────────────────────────────────────

Answer: According to Section 5 of the Service Agreement, there are
two types of termination conditions:

1. **Standard termination**: Either party may terminate with 30 days
   written notice.

2. **Immediate termination**: Permitted in cases of:
   - Material breach of the agreement
   - Bankruptcy of either party
   - Failure to pay fees within 60 days

Sources:
[1] Service Agreement > 5. Termination

──────────────────────────────────────────────────────────────
📊 Confidence: 0.89 | Sources: 3
```

## 🎯 Use Cases

| Domain | Application |
|--------|-------------|
| **Legal** | Contract analysis, clause extraction, cross-reference tracking |
| **Technical** | Documentation search, API reference, specification lookup |
| **Medical** | Clinical guidelines, drug interactions, protocol search |
| **Academic** | Literature review, citation tracking, research synthesis |
| **Corporate** | Policy search, compliance checking, knowledge management |

## 🔬 Advanced Usage

### Custom Document Parsing

```python
from hierarchical_rag import DocumentParser, Section, Paragraph

class CustomParser(DocumentParser):
    def parse_legal_document(self, text: str):
        # Custom logic for legal documents
        # e.g., detect "WHEREAS" clauses, numbered articles, etc.
        pass
```

### Embedding Integration

Replace the demo embeddings with production embeddings:

```python
# In rag.py, modify _get_embedding():
def _get_embedding(self, text: str) -> list[float]:
    # Use Voyage AI
    response = voyageai.Client().embed(text, model="voyage-3")
    return response.embeddings[0]
    
    # Or use OpenAI
    response = openai.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding
```

## 🛡️ Limitations

- Embedding quality depends on the model used (demo uses simple hashing)
- Very large documents may need chunking strategies
- Cross-reference resolution is pattern-based (not semantic)
- Summary generation adds latency on first index build

## 📄 License

MIT License - feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional document format parsers (PDF, DOCX)
- Improved cross-reference patterns
- Caching strategies
- Streaming responses


