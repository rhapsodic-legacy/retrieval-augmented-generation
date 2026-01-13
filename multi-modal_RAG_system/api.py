"""
FastAPI Backend for Multi-Modal RAG

Provides REST API for:
- File upload and indexing
- Multi-modal search
- Query with image context
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional
import json
import os
import base64
import tempfile
from pathlib import Path

from multimodal_rag import MultimodalRAG, MultimodalRAGConfig


app = FastAPI(
    title="Multi-Modal RAG API",
    description="RAG system handling text, images, PDFs, and tables",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_instance: Optional[MultimodalRAG] = None


class ConfigRequest(BaseModel):
    llm_provider: str = "anthropic"
    llm_model: Optional[str] = None
    embedding_provider: str = "local"
    n_results: int = 5


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    include_images: bool = True


class TextRequest(BaseModel):
    text: str
    source: str = "text"


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Multi-Modal RAG API</h1><p>Visit /docs for API documentation</p>"


@app.post("/api/init")
async def initialize(config: ConfigRequest):
    global rag_instance
    
    try:
        rag_config = MultimodalRAGConfig(
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            embedding_provider=config.embedding_provider,
            n_results=config.n_results,
        )
        
        rag_instance = MultimodalRAG(rag_config)
        
        return {"status": "initialized", "config": config.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    if not rag_instance:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready",
        "stats": rag_instance.get_stats(),
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    # Save to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        stats = rag_instance.add_file(tmp_path)
        return {
            "status": "indexed",
            "filename": file.filename,
            "stats": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/add/text")
async def add_text(req: TextRequest):
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    stats = rag_instance.add_text(req.text, req.source)
    return {"status": "indexed", "stats": stats}


@app.post("/api/query")
async def query(req: QueryRequest):
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        response = rag_instance.query(
            req.question,
            k=req.k,
            include_images=req.include_images,
        )
        
        # Format response
        return {
            "answer": response.answer,
            "sources": response.context.sources,
            "text_results": [
                {
                    "id": r.item.id,
                    "text": r.item.text[:500],
                    "source": r.item.source_file,
                    "score": r.score,
                }
                for r in response.context.text_results
            ],
            "image_results": [
                {
                    "id": r.item.id,
                    "description": r.item.text[:300],
                    "source": r.item.source_file,
                    "type": r.item.image_type if hasattr(r.item, 'image_type') else "unknown",
                    "score": r.score,
                    "has_data": r.item.image_data is not None if hasattr(r.item, 'image_data') else False,
                }
                for r in response.context.image_results
            ],
            "table_results": [
                {
                    "id": r.item.id,
                    "caption": r.item.caption if hasattr(r.item, 'caption') else None,
                    "columns": r.item.headers if hasattr(r.item, 'headers') else [],
                    "num_rows": len(r.item.rows) if hasattr(r.item, 'rows') else 0,
                    "source": r.item.source_file,
                    "score": r.score,
                }
                for r in response.context.table_results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    item = rag_instance.index.get(image_id)
    if not item or not hasattr(item, 'image_data') or not item.image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return Response(
        content=item.image_data,
        media_type=f"image/{item.format}" if hasattr(item, 'format') else "image/jpeg",
    )


@app.get("/api/table/{table_id}")
async def get_table(table_id: str):
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    item = rag_instance.index.get(table_id)
    if not item or not hasattr(item, 'headers'):
        raise HTTPException(status_code=404, detail="Table not found")
    
    return {
        "id": item.id,
        "caption": item.caption if hasattr(item, 'caption') else None,
        "headers": item.headers,
        "rows": item.rows[:100],  # Limit rows
        "total_rows": len(item.rows),
        "markdown": item.to_markdown() if hasattr(item, 'to_markdown') else None,
    }


@app.post("/api/search/images")
async def search_images(req: QueryRequest):
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    images = rag_instance.find_images(req.question, k=req.k)
    
    return {
        "results": [
            {
                "id": img.id,
                "description": img.description or img.text,
                "type": img.image_type,
                "source": img.source_file,
            }
            for img in images
        ]
    }


@app.post("/api/search/tables")
async def search_tables(req: QueryRequest):
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    tables = rag_instance.find_tables(req.question, k=req.k)
    
    return {
        "results": [
            {
                "id": t.id,
                "caption": t.caption,
                "columns": t.headers,
                "num_rows": len(t.rows),
                "source": t.source_file,
            }
            for t in tables
        ]
    }


@app.get("/api/sources")
async def list_sources():
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    return {"sources": rag_instance.list_sources()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
