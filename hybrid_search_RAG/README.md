# 🔍 Hybrid Search RAG

Production-grade RAG combining **Vector Search**, **BM25**, **Knowledge Graphs**, and **ML Fusion**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![React](https://img.shields.io/badge/React-18-blue)

## ✨ Features

| Component | Description |
|-----------|-------------|
| 🎯 **Vector Search** | Semantic similarity using dense embeddings (ChromaDB, FAISS) |
| 📝 **BM25 Search** | Keyword matching for exact terms, names, codes |
| 🔗 **Knowledge Graph** | Entity relationships and graph traversal |
| 🤖 **ML Fusion** | Learned re-ranking combining all signals |
| 🌐 **Web UI** | React frontend with graph visualization |
| 🔌 **Multi-Provider** | Claude, Gemini, GPT + multiple embedding options |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/hybrid-search-rag.git
cd hybrid-search-rag

pip install -r requirements.txt

# Set API key (choose one)
export ANTHROPIC_API_KEY="your-key"  # Claude
export GOOGLE_API_KEY="your-key"     # Gemini
export OPENAI_API_KEY="your-key"     # OpenAI
```

### Start Web UI

```bash
python main.py serve
# Open http://localhost:8000
```

### Python Usage

```python
from hybrid_rag import HybridRAG, RAGConfig

# Configure with your preferred providers
config = RAGConfig(
    llm_provider="anthropic",      # or "gemini", "openai"
    embedding_provider="local",     # or "voyage", "openai", "google"
    fusion_strategy="rrf",          # or "weighted", "learned"
)

rag = HybridRAG(config)

# Add documents
rag.add_document("doc1", "Machine learning is a subset of AI...", title="ML Intro")
rag.add_document("doc2", "Neural networks use layers of nodes...", title="Neural Nets")

# Query with hybrid search
response = rag.query("What is machine learning?")
print(response.answer)

# See which methods found each result
for source in response.sources:
    print(f"{source.id}: {source.sources}")  # ['vector', 'sparse']
```

### CLI Usage

```bash
# Interactive mode
python main.py --provider anthropic interactive --file docs/

# Single query
python main.py query "What is X?" --file document.md

# Compare search methods
python main.py compare "neural networks" --file docs/

# Demo with sample data
python main.py demo
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         QUERY                               │
│                  "What is deep learning?"                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  🎯 VECTOR      │ │  📝 BM25        │ │  🔗 GRAPH       │
│                 │ │                 │ │                 │
│ Semantic search │ │ Keyword match   │ │ Entity lookup   │
│ via embeddings  │ │ TF-IDF scoring  │ │ Relationship    │
│                 │ │                 │ │ traversal       │
│ Score: 0.85     │ │ Score: 12.4     │ │ Score: 0.67     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      🔄 FUSION                              │
│                                                             │
│  RRF: 1/(k+rank) summation across all methods              │
│  Weighted: α·vector + β·sparse + γ·graph                   │
│  Learned: ML model predicts relevance from features        │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      🤖 LLM                                 │
│                                                             │
│  Generate answer using top-k fused results as context      │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
hybrid-search-rag/
├── hybrid_rag/
│   ├── __init__.py
│   ├── rag.py                 # Main HybridRAG class
│   ├── providers/
│   │   ├── llm.py            # Claude, Gemini, OpenAI
│   │   └── embeddings.py     # Voyage, OpenAI, Google, local
│   ├── search/
│   │   ├── vector.py         # ChromaDB, FAISS, in-memory
│   │   ├── sparse.py         # BM25, TF-IDF
│   │   ├── knowledge_graph.py # Entity graph
│   │   └── hybrid.py         # Combined search
│   └── fusion/
│       └── strategies.py     # RRF, Weighted, Learned
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React UI
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Full Configuration Options

```python
from hybrid_rag import HybridRAG, RAGConfig

config = RAGConfig(
    # LLM Provider
    llm_provider="anthropic",    # "anthropic", "gemini", "openai"
    llm_model=None,              # Uses default if None
    
    # Embedding Provider
    embedding_provider="local",   # "voyage", "openai", "google", 
                                  # "sentence_transformers", "local"
    embedding_model=None,
    
    # Search Methods (enable/disable)
    enable_vector=True,
    enable_sparse=True,
    enable_graph=True,
    
    # Fusion Strategy
    fusion_strategy="rrf",       # "rrf", "weighted", "learned"
    vector_weight=0.4,           # For weighted fusion
    sparse_weight=0.35,
    graph_weight=0.25,
    
    # Retrieval
    n_results=5,
    
    # Generation
    max_tokens=1000,
    temperature=0.1,
)

rag = HybridRAG(config)
```

### Provider Comparison

| LLM Provider | Model | Best For |
|--------------|-------|----------|
| Anthropic | Claude 3.5 Sonnet | Complex reasoning, safety |
| Google | Gemini 1.5 Pro | Long context, speed |
| OpenAI | GPT-4o | General purpose |

| Embedding Provider | Model | Dimension | Best For |
|-------------------|-------|-----------|----------|
| Voyage | voyage-3 | 1024 | Claude users |
| OpenAI | text-embedding-3-small | 1536 | OpenAI users |
| Google | text-embedding-004 | 768 | Gemini users |
| Sentence Transformers | all-MiniLM-L6-v2 | 384 | Local/free |

## 🔄 Fusion Strategies

### Reciprocal Rank Fusion (RRF)
Best for most cases. Doesn't require score normalization.

```python
# Score = Σ 1/(k + rank) for each method
config = RAGConfig(fusion_strategy="rrf")
```

### Weighted Combination
When you know relative importance of each method.

```python
config = RAGConfig(
    fusion_strategy="weighted",
    vector_weight=0.5,   # Semantic understanding
    sparse_weight=0.3,   # Exact matches
    graph_weight=0.2,    # Relationships
)
```

### Learned Fusion
Train an ML model on your data for best results.

```python
from hybrid_rag.fusion import LearnedFusion

fusion = LearnedFusion()
fusion.train(training_data, save_path="model.pkl")

config = RAGConfig(fusion_strategy="learned")
rag = HybridRAG(config)
```

## 🔗 Knowledge Graph

### Adding Entities and Relations

```python
# Add custom entities
rag.add_entity(
    entity_id="person_1",
    name="John Smith",
    entity_type="person",
    properties={"role": "CEO"}
)

# Add relations
rag.add_relation(
    source_id="person_1",
    target_id="company_1",
    relation_type="works_for"
)

# Query related entities
related = rag.get_related_entities("person_1", max_depth=2)
```

### Entity Types
- `person`, `organization`, `location`
- `concept`, `product`, `event`
- `document`, `section`

### Relation Types
- `mentions`, `contains`, `related_to`
- `part_of`, `works_for`, `located_in`
- `created_by`, `references`

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/init` | Initialize system with config |
| GET | `/api/status` | Get system status |
| POST | `/api/documents` | Add a document |
| GET | `/api/documents` | List all documents |
| POST | `/api/query` | Query with hybrid search |
| POST | `/api/search/compare` | Compare search methods |
| GET | `/api/graph` | Get knowledge graph for visualization |

### Example API Calls

```bash
# Initialize
curl -X POST http://localhost:8000/api/init \
  -H "Content-Type: application/json" \
  -d '{"llm_provider": "anthropic", "fusion_strategy": "rrf"}'

# Add document
curl -X POST http://localhost:8000/api/documents \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "content": "...", "title": "My Document"}'

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "k": 5}'
```

## 📊 When to Use Each Search Method

| Method | Best For | Example Queries |
|--------|----------|-----------------|
| **Vector** | Semantic meaning, concepts | "How do neural networks learn?" |
| **BM25** | Exact terms, names, codes | "Error code E-401", "John Smith" |
| **Graph** | Relationships, connections | "What's related to X?", entity queries |
| **Hybrid** | General queries | Most real-world queries |

## 🎨 Web UI Features

The React frontend includes:

- **Configuration Panel**: Set LLM, embeddings, and fusion strategy
- **Document Management**: Add/view/delete documents
- **Query Interface**: Search with results breakdown
- **Knowledge Graph Viewer**: Interactive vis.js visualization
- **Score Comparison**: See how each method scored results

## 📈 Performance Tips

1. **Use RRF for most cases** - It's simple and works well without tuning
2. **Enable all methods** - They complement each other
3. **BM25 for exact matches** - Names, codes, technical terms
4. **Vector for concepts** - "What is...", "How does...", "Explain..."
5. **Graph for relationships** - When entities and connections matter
6. **Train learned fusion** - If you have relevance labels

## 🔬 Comparing Search Methods

```python
# See what each method finds
comparison = rag.compare_search_methods("neural networks", k=10)

print(f"Vector found: {comparison['vector']['ids']}")
print(f"BM25 found: {comparison['sparse']['ids']}")
print(f"Overlap: {comparison['overlap']}")
print(f"Vector-only: {comparison['vector']['unique']}")
print(f"BM25-only: {comparison['sparse']['unique']}")
```

