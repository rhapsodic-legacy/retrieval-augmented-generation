# ⚡ Real-Time Streaming RAG

RAG over continuously updating data with **incremental indexing**, **time-weighted retrieval**, **deduplication**, and **freshness-aware answers**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![WebSocket](https://img.shields.io/badge/WebSocket-Live-orange)
![React](https://img.shields.io/badge/React-18-blue)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Incremental Updates** | Add/remove content without full index rebuild |
| ⏱️ **Time-Weighted Retrieval** | Recent content ranked higher (configurable decay) |
| 🔍 **Deduplication** | MinHash LSH detects near-duplicate content |
| 🕐 **Freshness-Aware** | Answers include "As of 2 hours ago..." |
| 📡 **Live Streaming** | WebSocket for real-time updates |
| 🎨 **Eye-Catching UI** | Modern dark theme with animations |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/streaming-rag.git
cd streaming-rag

pip install -r requirements.txt

# Set API key for your chosen provider:
export GOOGLE_API_KEY="your-key"     # For Gemini (recommended - free tier)
# OR
export ANTHROPIC_API_KEY="your-key"  # For Claude
# OR
export OPENAI_API_KEY="your-key"     # For OpenAI
```

### Start Web UI

```bash
python main.py serve
# Open http://localhost:8000
```

### Demo Mode

```bash
# Start demo with simulated news and social streams
python main.py demo
```

### Python Usage

```python
from streaming_rag import StreamingRAG, StreamingRAGConfig

# Initialize with your preferred provider
config = StreamingRAGConfig(
    llm_provider="gemini",    # Options: "gemini", "anthropic", "openai"
    time_decay_hours=24,      # Half-life for relevance
)
rag = StreamingRAG(config)

# Or use a different provider:
# config = StreamingRAGConfig(llm_provider="anthropic")
# config = StreamingRAGConfig(llm_provider="openai")

# Start live streams
rag.start_demo_streams()

# Query with freshness awareness
response = rag.query("What's the latest on AI?")
print(response.answer)
print(f"Freshness: {response.freshness_note}")
# Output: "As of 5 minutes ago..."

# Add custom data
rag.add_item(
    content="Breaking: New AI breakthrough announced",
    content_type="news",
    source="my-feed",
)

# Get recent items
recent = rag.get_recent(10)

# Stop streams
rag.stop_streams()
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                             │
│         RSS Feeds • Webhooks • Log Files • APIs             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    INGESTORS                                │
│                                                             │
│  RSSIngestor → polls feeds                                  │
│  WebhookIngestor → receives HTTP posts                      │
│  LogFileIngestor → tails log files                          │
│  SimulatedIngestor → generates demo data                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                DEDUPLICATION ENGINE                         │
│                                                             │
│  Exact Hash → catches identical content                     │
│  MinHash LSH → catches near-duplicates (configurable)       │
│  Marks duplicates but keeps for comparison                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TIME-WEIGHTED STREAMING INDEX                  │
│                                                             │
│  Incremental add/remove (no rebuild)                        │
│  Time decay scoring (exponential/linear/log)                │
│  Priority weighting (critical > low)                        │
│  Automatic cleanup of old content                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              FRESHNESS-AWARE GENERATION                     │
│                                                             │
│  "As of [time], the latest information shows..."            │
│  Prioritizes recent sources                                 │
│  Notes rapidly changing information                         │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
streaming-rag/
├── streaming_rag/
│   ├── __init__.py
│   ├── rag.py                  # Main orchestrator
│   ├── providers/
│   │   ├── llm.py             # Claude, Gemini, GPT
│   │   └── embeddings.py      # Voyage, OpenAI, local
│   ├── ingestion/
│   │   ├── content_types.py   # StreamItem, NewsItem, etc.
│   │   └── ingestors.py       # RSS, Webhook, Log, Simulated
│   ├── indexing/
│   │   └── streaming_index.py # Time-weighted incremental index
│   └── dedup/
│       └── dedup_engine.py    # MinHash LSH deduplication
├── api.py                      # FastAPI + WebSocket
├── frontend/
│   └── index.html             # Eye-catching React UI
├── main.py                     # CLI
├── requirements.txt
└── README.md
```

## 🔧 Configuration

```python
from streaming_rag import StreamingRAG, StreamingRAGConfig

config = StreamingRAGConfig(
    # LLM Provider - choose one:
    # - "gemini": Google Gemini (free tier available, fast)
    # - "anthropic": Claude (excellent quality)
    # - "openai": GPT-4o-mini (good balance)
    llm_provider="gemini",
    llm_model=None,  # Uses provider default
    
    # Embeddings
    embedding_provider="local",  # "voyage" or "openai" for production
    
    # Time decay
    time_decay_hours=24,  # Half-life for relevance score
    max_age_days=7,       # Auto-cleanup after this
    
    # Deduplication
    enable_dedup=True,
    dedup_threshold=0.8,  # Similarity threshold
    
    # Search
    n_results=10,
    time_weight=0.3,      # How much to weight recency
    priority_weight=0.1,  # How much to weight priority
    
    # Generation
    max_tokens=2000,
    temperature=0.1,
)
```

### Supported LLM Providers

| Provider | Model | API Key Env Var | Notes |
|----------|-------|-----------------|-------|
| `gemini` | gemini-1.5-flash | `GOOGLE_API_KEY` | Free tier available, fast |
| `anthropic` | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` | Excellent quality |
| `openai` | gpt-4o-mini | `OPENAI_API_KEY` | Good balance |

## ⏱️ Time-Weighted Scoring

Final score combines semantic relevance, time decay, and priority:

```
final_score = semantic_score × 0.6 + time_score × 0.3 + priority_score × 0.1
```

### Time Decay Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **Exponential** | e^(-λt) | News, social media (fast decay) |
| **Linear** | 1 - t/max | Logs, reports (steady decay) |
| **Logarithmic** | 1/(1 + log(1+t)) | Research, docs (slow decay) |

### Freshness Boost

Content within the first hour gets a 1.5× boost to ensure breaking news surfaces.

## 🔍 Deduplication

### MinHash LSH

Uses Locality-Sensitive Hashing for fast near-duplicate detection:

1. **Shingling**: Convert text to word n-grams
2. **MinHash**: Generate signature (128 hashes)
3. **LSH Banding**: Hash bands to buckets
4. **Candidate Check**: Verify similarity of candidates

### Configuration

```python
from streaming_rag.dedup import DedupConfig

config = DedupConfig(
    num_hashes=128,
    shingle_size=3,
    num_bands=16,
    similarity_threshold=0.8,  # 80% similar = duplicate
)
```

## 📡 Data Sources

### RSS Feeds

```python
rag.add_rss_feed(
    "https://news.ycombinator.com/rss",
    source_name="hackernews"
)
```

### Webhooks

```bash
curl -X POST http://localhost:8000/api/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Breaking news...",
    "type": "news",
    "source": "my-feed"
  }'
```

### Manual Items

```python
rag.add_item(
    content="Important update...",
    content_type="alert",
    source="internal",
    priority=ContentPriority.HIGH,
)
```

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/init` | Initialize system |
| POST | `/api/streams/demo/start` | Start demo streams |
| POST | `/api/streams/stop` | Stop all streams |
| POST | `/api/items` | Add an item |
| POST | `/api/webhook` | Webhook ingestion |
| POST | `/api/query` | Query with freshness |
| GET | `/api/recent` | Get recent items |
| WS | `/ws` | Real-time updates |

### WebSocket Events

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8000/ws');

// Receive new items
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'new_item') {
        console.log('New:', data.data);
    }
};

// Request recent items
ws.send(JSON.stringify({ type: 'get_recent', n: 10 }));
```

## 🎨 Web UI Features

- **Dark Theme**: Modern gradient design
- **Live Stream Feed**: Real-time updates with animations
- **Score Visualization**: See semantic, time, and final scores
- **Freshness Labels**: "Just now", "5 minutes ago", etc.
- **Type Badges**: Color-coded news, social, logs, alerts
- **WebSocket Status**: Live connection indicator

## 📊 Content Types

| Type | Description | Priority |
|------|-------------|----------|
| `news` | News articles | Based on source |
| `social` | Social media posts | Based on engagement |
| `log` | System logs | Based on level (ERROR > INFO) |
| `alert` | Alerts/notifications | Typically HIGH |
| `generic` | Other content | MEDIUM |

## 💡 Best Practices

1. **Set appropriate time decay** - News: 6-12h, Logs: 24-48h
2. **Use deduplication** - Especially for RSS feeds
3. **Monitor duplicates** - High rate may indicate source issues
4. **Configure cleanup** - Prevent unbounded growth
5. **Use priority levels** - Ensure critical content surfaces

## 📈 Performance Tips

- **Batch ingestion** when possible
- **Use local embeddings** for high-volume streams
- **Set max_items limit** for memory management
- **Enable cleanup thread** for long-running services
- **Monitor WebSocket connections**

## 🔬 Example Queries

| Query | What It Does |
|-------|-------------|
| "What's the latest on AI?" | Time-weighted search, recent prioritized |
| "Any breaking news?" | Searches news type, very recent |
| "Error logs from today" | Filters by type and time range |
| "Trending topics" | Social content, engagement-weighted |

## 📝 License

MIT License
