# 🖼️ Multi-Modal RAG System

A unified RAG system that handles **text, images, PDFs, and tables** together with cross-modal search.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![React](https://img.shields.io/badge/React-18-blue)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Extraction** | Layout-aware text, images, and tables from PDFs |
| 🖼️ **Image Processing** | Vision model descriptions, chart data extraction |
| 📊 **Table Handling** | Structured data, not flattened text |
| 🔍 **Cross-Modal Search** | "Find the chart showing Q3 revenue" |
| 🤖 **Vision LLMs** | Claude, Gemini, GPT-4o with image understanding |
| 🌐 **Web UI** | React frontend for upload and search |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag

pip install -r requirements.txt

# Set API key (need vision-capable model)
export ANTHROPIC_API_KEY="your-key"  # Best for vision
# or GOOGLE_API_KEY, OPENAI_API_KEY
```

### Start Web UI

```bash
python main.py serve
# Open http://localhost:8000
```

### Python Usage

```python
from multimodal_rag import MultimodalRAG, MultimodalRAGConfig

# Initialize with vision-capable LLM
config = MultimodalRAGConfig(
    llm_provider="anthropic",      # Claude has excellent vision
    embedding_provider="local",
)
rag = MultimodalRAG(config)

# Add content from different modalities
rag.add_pdf("quarterly_report.pdf")    # Extracts text, images, tables
rag.add_image("revenue_chart.png")      # Describes and embeds image
rag.add_table("sales_data.csv")         # Structured table handling

# Query across all modalities
response = rag.query("Find the chart showing Q3 revenue")
print(response.answer)

# Access images and tables in response
for img in response.images:
    print(f"Found: {img.description}")
    # img.image_data contains the raw bytes

for table in response.tables:
    print(f"Table: {table.caption}")
    print(table.to_markdown())
```

### CLI Usage

```bash
# Interactive mode
python main.py interactive report.pdf data.csv

# Single query
python main.py query "What are the Q3 results?" -f report.pdf

# Search specific modalities
python main.py interactive
> /images revenue chart
> /tables sales data
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT CONTENT                            │
│         PDF • Images • CSV/Excel • HTML • Text              │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  PDF EXTRACTOR  │ │ IMAGE PROCESSOR │ │ TABLE EXTRACTOR │
│                 │ │                 │ │                 │
│ • Layout-aware  │ │ • Vision LLM    │ │ • CSV/Excel     │
│ • Text blocks   │ │ • Description   │ │ • HTML tables   │
│ • Images        │ │ • Classification│ │ • Structured    │
│ • Tables        │ │ • Chart data    │ │ • Queryable     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 UNIFIED MULTIMODAL INDEX                    │
│                                                             │
│  All content embedded in same vector space                  │
│  Images: description embeddings                             │
│  Tables: text + structure embeddings                        │
│  Text: standard text embeddings                             │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    CROSS-MODAL SEARCH                       │
│                                                             │
│  Query: "Find the chart showing Q3 revenue"                 │
│  → Searches text, images, and tables simultaneously         │
│  → Returns relevant content from all modalities             │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   VISION-AWARE LLM                          │
│                                                             │
│  Receives: query + text context + actual images             │
│  Generates: comprehensive answer with visual understanding  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
multimodal-rag/
├── multimodal_rag/
│   ├── __init__.py
│   ├── rag.py                    # Main orchestrator
│   ├── providers/
│   │   ├── llm.py               # Vision LLMs (Claude, Gemini, GPT-4o)
│   │   └── embeddings.py        # Text + multimodal embeddings
│   ├── extraction/
│   │   ├── content_types.py     # ContentItem, ImageContent, TableContent
│   │   ├── pdf_extractor.py     # Layout-aware PDF extraction
│   │   ├── image_processor.py   # Vision model image processing
│   │   └── table_extractor.py   # CSV, Excel, HTML, Markdown tables
│   └── indexing/
│       └── multimodal_index.py  # Unified cross-modal index
├── api.py                        # FastAPI backend
├── frontend/
│   └── index.html               # React UI
├── main.py                       # CLI
├── requirements.txt
└── README.md
```

## 🔧 Configuration

```python
from multimodal_rag import MultimodalRAG, MultimodalRAGConfig

config = MultimodalRAGConfig(
    # LLM (must support vision for image processing)
    llm_provider="anthropic",    # "anthropic", "gemini", "openai"
    llm_model=None,              # Uses default if None
    
    # Embeddings
    embedding_provider="local",   # "voyage", "openai", "google", "local"
    embedding_model=None,
    
    # PDF extraction
    extract_pdf_images=True,      # Extract images from PDFs
    extract_pdf_tables=True,      # Extract tables from PDFs
    pdf_chunk_size=1000,          # Text chunk size
    
    # Image processing
    generate_image_descriptions=True,   # Use vision LLM to describe
    extract_chart_data=True,            # Parse charts for data
    
    # Search
    n_results=5,
    
    # Generation
    max_tokens=2000,
    temperature=0.1,
)

rag = MultimodalRAG(config)
```

## 📄 PDF Extraction

### Layout-Aware Processing

```python
# Extracts with layout preservation
rag.add_pdf("report.pdf")

# What gets extracted:
# - Text blocks (preserving paragraph structure)
# - Images (with bounding boxes)
# - Tables (as structured data)
```

### Using pdfplumber + PyMuPDF

| Library | Used For |
|---------|----------|
| pdfplumber | Text extraction with positions, table detection |
| PyMuPDF | Image extraction, page rendering |

## 🖼️ Image Processing

### Vision Model Processing

```python
# Images are processed with the vision LLM:
# 1. Classification (photo, chart, diagram, etc.)
# 2. Description generation
# 3. Chart data extraction (if applicable)
# 4. OCR for text in images

image = rag.find_images("revenue chart")[0]
print(image.description)     # "Bar chart showing quarterly revenue..."
print(image.image_type)      # "chart"
print(image.chart_type)      # "bar"
print(image.data_points)     # [{"label": "Q1", "value": 1.2M}, ...]
```

### Image Types Detected

- `photo` - Real-world photographs
- `chart` - Bar, line, pie charts
- `diagram` - Flow diagrams, schematics
- `screenshot` - Software screenshots
- `document` - Scanned documents
- `table` - Images of tables

## 📊 Table Handling

### Structured, Not Flattened

```python
# Tables are stored as structured data
rag.add_table("sales.csv")

# Query the table
tables = rag.find_tables("sales by region")
table = tables[0]

print(table.headers)      # ["Region", "Q1", "Q2", "Q3", "Q4"]
print(table.rows[0])      # ["North", 100, 120, 140, 160]

# Get as markdown
print(table.to_markdown())

# Query specific data
results = table.query("Region", "North")
```

### Supported Formats

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| TSV | `.tsv` |
| Excel | `.xlsx`, `.xls` |
| HTML | `<table>` elements |
| Markdown | `\| col \|` tables |

## 🔍 Cross-Modal Search

### Query Examples

| Query | What It Finds |
|-------|--------------|
| "Q3 revenue" | Text mentions + revenue charts + financial tables |
| "product comparison" | Comparison charts + tables + text sections |
| "team photo" | Photos of people/teams |
| "architecture diagram" | Technical diagrams |
| "sales by region" | Regional data in tables + charts |

### How It Works

1. **Query embedding** - Query converted to vector
2. **Parallel search** - Text, images, tables searched simultaneously
3. **Score fusion** - Results combined and ranked
4. **Context building** - Best results from each modality
5. **Vision LLM** - Answer generated with image context

## 🌐 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/init` | Initialize system |
| POST | `/api/upload` | Upload any file |
| POST | `/api/query` | Cross-modal query |
| GET | `/api/image/{id}` | Get image by ID |
| GET | `/api/table/{id}` | Get table data |
| POST | `/api/search/images` | Search images only |
| POST | `/api/search/tables` | Search tables only |

### Example API Calls

```bash
# Initialize
curl -X POST http://localhost:8000/api/init \
  -H "Content-Type: application/json" \
  -d '{"llm_provider": "anthropic"}'

# Upload PDF
curl -X POST http://localhost:8000/api/upload \
  -F "file=@report.pdf"

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Find the Q3 revenue chart", "k": 5}'

# Get image
curl http://localhost:8000/api/image/{image_id} --output chart.png
```

## 🎨 Web UI Features

- **Drag & Drop Upload**: PDF, images, CSV, Excel
- **Cross-Modal Search**: Single query searches everything
- **Result Categories**: Text, images, tables shown separately
- **Image Preview**: View images inline
- **Table Display**: See table structure and data

## 💡 Best Practices

1. **Use a flagship LLM for vision** - Most consistent image understanding
2. **Extract tables from PDFs** - Better than OCR on table images
3. **Enable chart data extraction** - Get actual numbers from charts
4. **Query specific modalities** - `/images` or `/tables` for targeted search
5. **Check image descriptions** - Verify quality for your domain

## 📈 Performance Tips

- **Pre-process large PDFs** - Index once, query many times
- **Use local embeddings for testing** - API embeddings for production
- **Limit image processing** - Vision API calls are expensive
- **Chunk tables** - Large tables are split for better retrieval

## 🔬 Content Types

```python
from multimodal_rag import ContentType, ImageContent, TableContent

# Content types in the system
ContentType.TEXT      # Text chunks
ContentType.IMAGE     # Images with descriptions
ContentType.TABLE     # Structured tables
ContentType.CHART     # Charts with extracted data
ContentType.PDF_PAGE  # Full PDF pages
```

## 📝 License

MIT License
