# 💬 Conversational RAG with Memory

Production-ready conversational RAG with persistent memory, supporting both **Claude** and **Gemini** as LLM backends.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Claude](https://img.shields.io/badge/Claude-Supported-orange)
![Gemini](https://img.shields.io/badge/Gemini-Supported-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Query Rewriting** | Automatically resolves pronouns and references ("the second one" → actual entity) |
| 💾 **Session Memory** | Tracks current conversation context, entities, and citations |
| 🧠 **Long-Term Memory** | Persists user facts, preferences, and document knowledge across sessions |
| 📚 **Citation Tracking** | Maintains source references across multi-turn conversations |
| 🔌 **Multi-Provider** | Supports Claude (Anthropic) and Gemini (Google) |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/conversational-rag.git
cd conversational-rag

pip install -r requirements.txt

# Set your API key (choose one)
export ANTHROPIC_API_KEY="your-key"  # For Claude
export GOOGLE_API_KEY="your-key"     # For Gemini
```

### Basic Usage

```python
from conversational_rag import ConversationalRAG, ConversationConfig

# Using Claude (default)
config = ConversationConfig(llm_provider="anthropic")
rag = ConversationalRAG(config)

# Or using Gemini
config = ConversationConfig(llm_provider="gemini")
rag = ConversationalRAG(config)

# Add documents
rag.add_document("""
# Project Overview
Our project focuses on building scalable microservices.
The authentication service handles user login and JWT tokens.
The data service manages all database operations.
""", title="Project Docs")

# First question
response = rag.chat("What services are mentioned?")
print(response.answer)
# "The document mentions two services: the authentication service 
#  (handles login and JWT tokens) and the data service (manages 
#  database operations). [1]"

# Follow-up with pronoun
response = rag.chat("Tell me more about the first one")
print(response.answer)
# Automatically understands "first one" = "authentication service"

# Reference previous answer
response = rag.chat("How does it handle security?")
print(response.answer)
# Maintains context that "it" refers to authentication service
```

### Command Line

```bash
# Interactive chat with Claude
python main.py --provider anthropic interactive --file docs/report.md

# Interactive chat with Gemini
python main.py --provider gemini interactive --file docs/report.md

# With persistent memory
python main.py --storage ./memory --user john interactive

# Demo mode
python main.py demo
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER MESSAGE                                │
│              "Tell me more about the second feature"                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY REWRITER                                 │
│  • Checks conversation history                                      │
│  • Resolves "the second feature" → "Habit Tracking"                │
│  • Rewrites: "Tell me more about the Habit Tracking feature"       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL                                      │
│  • Vector search with rewritten query                               │
│  • Returns relevant chunks with metadata                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CITATION TRACKER                                 │
│  • Assigns citation IDs [1], [2], etc.                             │
│  • Tracks sources across conversation                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                                   │
│  • Claude or Gemini                                                 │
│  • Includes conversation history                                    │
│  • Generates cited answer                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MEMORY UPDATE                                    │
│  • Session: Updates conversation, entities, citations               │
│  • Long-term: Extracts facts (periodic)                            │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
conversational-rag/
├── conversational_rag/
│   ├── __init__.py           # Package exports
│   ├── rag.py                # Main ConversationalRAG class
│   ├── query_rewriter.py     # Pronoun/reference resolution
│   ├── citation_tracker.py   # Citation management
│   ├── providers/
│   │   ├── llm.py           # Claude & Gemini providers
│   │   └── embeddings.py    # Embedding providers
│   ├── memory/
│   │   ├── session.py       # Current conversation memory
│   │   ├── long_term.py     # Persistent user memory
│   │   └── manager.py       # Unified memory interface
│   └── core/
│       └── retriever.py     # Document store & search
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## 🔧 Configuration

```python
config = ConversationConfig(
    # LLM Settings
    llm_provider="anthropic",        # "anthropic" or "gemini"
    llm_model=None,                  # Uses default if None
    
    # Embedding Settings  
    embedding_provider="local",      # "voyage", "google", "openai", "local"
    
    # Memory Settings
    user_id="default_user",          # For memory isolation
    storage_path="./memory",         # Persistence directory
    auto_persist=True,               # Auto-save sessions
    
    # Retrieval Settings
    n_results=5,                     # Documents to retrieve
    chunk_size=500,                  # Chunk size for documents
    
    # Query Rewriting
    enable_rewriting=True,
    rewrite_mode="hybrid",           # "hybrid", "llm", "rules"
    
    # Citation Format
    citation_format="bracket",       # "bracket", "superscript", "footnote"
)
```

## 🔄 Query Rewriting Examples

| User Says | Understood As |
|-----------|---------------|
| "Tell me more about it" | "Tell me more about [last discussed topic]" |
| "What about the second one?" | "What about [second item in list]?" |
| "How does this relate?" | "How does [current topic] relate to [previous topic]?" |
| "Continue" | Continues from last assistant response |
| "The previous answer mentioned X" | Retrieves context from previous turn |

## 💾 Memory System

### Session Memory (Per Conversation)
- Conversation history
- Active citations  
- Entity tracking (for pronoun resolution)
- Current topic

### Long-Term Memory (Persistent)
- User facts ("User prefers concise answers")
- Preferences ("citation_style": "academic")
- Document memory (titles, summaries, topics)
- Session summaries

```python
# Explicit memory operations
rag.remember("User is a software engineer", category="context")
rag.set_preference("response_length", "concise")

# Memory is also extracted automatically from conversations
```

## 🔌 Provider Comparison

| Feature | Claude (Anthropic) | Gemini (Google) |
|---------|-------------------|-----------------|
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Speed | Fast | Very Fast |
| Context Window | 200K | 1M+ |
| Best For | Complex reasoning | Long documents |
| Recommended Embeddings | Voyage | Google |

## 📊 Example Session

```
You: What authentication methods are supported?