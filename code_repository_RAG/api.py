"""
FastAPI Backend for Code RAG

Provides REST API for:
- Indexing codebases
- Code search and queries
- Symbol lookup
- Code flow tracing
- Dependency graph
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import os
from pathlib import Path
import tempfile
import zipfile

from code_rag import CodeRAG, CodeRAGConfig


app = FastAPI(
    title="Code RAG API",
    description="Intelligent Q&A over codebases with AST-aware parsing and dependency analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_instance: Optional[CodeRAG] = None


# =============================================================================
# Pydantic Models
# =============================================================================

class ConfigRequest(BaseModel):
    llm_provider: str = "anthropic"
    llm_model: Optional[str] = None
    embedding_provider: str = "local"
    n_results: int = 10
    include_tests: bool = True


class QueryRequest(BaseModel):
    question: str
    k: int = 10


class TraceRequest(BaseModel):
    start: str
    end: str


class SymbolRequest(BaseModel):
    name: str


# =============================================================================
# API Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Code RAG API</h1><p>Visit /docs for API documentation</p>"


@app.post("/api/init")
async def initialize(config: ConfigRequest):
    """Initialize the Code RAG system."""
    global rag_instance
    
    try:
        rag_config = CodeRAGConfig(
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            embedding_provider=config.embedding_provider,
            n_results=config.n_results,
            include_tests=config.include_tests,
        )
        
        rag_instance = CodeRAG(rag_config)
        
        return {
            "status": "initialized",
            "config": {
                "llm_provider": config.llm_provider,
                "embedding_provider": config.embedding_provider,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """Get system status."""
    if not rag_instance:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready",
        "stats": rag_instance.get_stats(),
    }


# =============================================================================
# Indexing
# =============================================================================

@app.post("/api/index/directory")
async def index_directory(path: str):
    """Index a local directory."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
    
    try:
        stats = rag_instance.index_directory(path)
        return {"status": "indexed", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index/upload")
async def index_upload(file: UploadFile = File(...)):
    """Upload and index a zip file containing code."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only zip files are supported")
    
    try:
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save and extract zip
            zip_path = Path(temp_dir) / "upload.zip"
            content = await file.read()
            zip_path.write_bytes(content)
            
            extract_dir = Path(temp_dir) / "code"
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Index
            stats = rag_instance.index_directory(str(extract_dir))
            
            return {"status": "indexed", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index/file")
async def index_file(file_path: str, content: str):
    """Index a single file with provided content."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        # Write to temp file and index
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=Path(file_path).suffix,
            delete=False,
        ) as f:
            f.write(content)
            temp_path = f.name
        
        result = rag_instance.index_file(temp_path)
        os.unlink(temp_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Query
# =============================================================================

@app.post("/api/query")
async def query(req: QueryRequest):
    """Query the codebase."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        response = rag_instance.query(req.question, k=req.k)
        
        return {
            "answer": response.answer,
            "sources": [
                {
                    "id": r.unit.id,
                    "name": r.unit.name,
                    "type": r.unit.type.value,
                    "file": r.unit.file_path,
                    "line": r.unit.start_line,
                    "signature": r.unit.signature,
                    "docstring": r.unit.docstring,
                    "score": r.score,
                }
                for r in response.context.primary_results
            ],
            "related": [
                {
                    "name": u.name,
                    "type": u.type.value,
                    "file": u.file_path,
                }
                for u in response.context.related_code[:5]
            ],
            "tests": [
                {
                    "name": u.name,
                    "file": u.file_path,
                }
                for u in response.context.test_code
            ],
            "files_involved": response.context.files_involved,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """Stream a query response."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    async def generate():
        try:
            for chunk in rag_instance.query_stream(req.question, k=req.k):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# =============================================================================
# Code Flow
# =============================================================================

@app.post("/api/trace")
async def trace_flow(req: TraceRequest):
    """Trace code flow between two points."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        response = rag_instance.trace_flow(req.start, req.end)
        
        return {
            "answer": response.answer,
            "flow": response.code_flow,
            "sources": [
                {
                    "name": r.unit.name,
                    "type": r.unit.type.value,
                    "file": r.unit.file_path,
                }
                for r in response.context.primary_results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Symbol Lookup
# =============================================================================

@app.post("/api/symbol/find")
async def find_symbol(req: SymbolRequest):
    """Find a symbol by name."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    return {"results": rag_instance.find_symbol(req.name)}


@app.post("/api/symbol/callers")
async def get_callers(req: SymbolRequest):
    """Get all callers of a function."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    return {"callers": rag_instance.get_callers(req.name)}


@app.post("/api/symbol/callees")
async def get_callees(req: SymbolRequest):
    """Get all functions called by a function."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    return {"callees": rag_instance.get_callees(req.name)}


@app.post("/api/symbol/tests")
async def get_tests(req: SymbolRequest):
    """Get tests for a function/class."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    return {"tests": rag_instance.get_tests_for(req.name)}


# =============================================================================
# Graph
# =============================================================================

@app.get("/api/graph")
async def get_graph():
    """Get the dependency graph for visualization."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    graph_data = rag_instance.graph.to_dict()
    
    # Format for visualization
    return {
        "nodes": [
            {
                "id": n["id"],
                "label": n["name"],
                "group": n["type"],
                "file": n.get("file_path"),
            }
            for n in graph_data["nodes"]
        ],
        "edges": [
            {
                "from": r["source_id"],
                "to": r["target_id"],
                "label": r["type"],
            }
            for r in graph_data["relationships"]
        ],
        "stats": rag_instance.graph.get_stats(),
    }


@app.get("/api/file/{file_path:path}")
async def get_file_summary(file_path: str):
    """Get a summary of a file."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    return rag_instance.get_file_summary(file_path)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
