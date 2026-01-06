"""
FastAPI Backend for Hybrid Search RAG

Provides REST API for:
- Document management
- Hybrid search
- Knowledge graph operations
- Query with RAG
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import os

from hybrid_rag import HybridRAG, RAGConfig


# Initialize FastAPI
app = FastAPI(
    title="Hybrid Search RAG API",
    description="API for hybrid search combining vector, BM25, and knowledge graph retrieval",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_instance: Optional[HybridRAG] = None


# =============================================================================
# Pydantic Models
# =============================================================================

class ConfigRequest(BaseModel):
    llm_provider: str = "anthropic"
    llm_model: Optional[str] = None
    embedding_provider: str = "local"
    embedding_model: Optional[str] = None
    enable_vector: bool = True
    enable_sparse: bool = True
    enable_graph: bool = True
    fusion_strategy: str = "rrf"


class DocumentRequest(BaseModel):
    id: str
    content: str
    title: Optional[str] = None
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    include_graph: bool = True


class EntityRequest(BaseModel):
    id: str
    name: str
    type: str = "other"
    properties: Optional[dict] = None


class RelationRequest(BaseModel):
    source_id: str
    target_id: str
    type: str = "related_to"
    weight: float = 1.0


# =============================================================================
# API Routes
# =============================================================================

@app.get("/")
async def root():
    """Serve the frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return {"message": "Hybrid Search RAG API", "docs": "/docs"}


@app.post("/api/init")
async def initialize(config: ConfigRequest):
    """Initialize or reinitialize the RAG system."""
    global rag_instance
    
    try:
        rag_config = RAGConfig(
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            enable_vector=config.enable_vector,
            enable_sparse=config.enable_sparse,
            enable_graph=config.enable_graph,
            fusion_strategy=config.fusion_strategy,
        )
        
        rag_instance = HybridRAG(rag_config)
        
        return {
            "status": "initialized",
            "config": {
                "llm_provider": config.llm_provider,
                "embedding_provider": config.embedding_provider,
                "fusion_strategy": config.fusion_strategy,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """Get system status and statistics."""
    if not rag_instance:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready",
        "stats": rag_instance.get_stats(),
    }


# =============================================================================
# Document Management
# =============================================================================

@app.post("/api/documents")
async def add_document(doc: DocumentRequest):
    """Add a document to the system."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized. Call /api/init first.")
    
    try:
        rag_instance.add_document(
            doc_id=doc.id,
            content=doc.content,
            title=doc.title,
            metadata=doc.metadata,
        )
        
        return {
            "status": "added",
            "id": doc.id,
            "stats": rag_instance.get_stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document file."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        content = await file.read()
        content_str = content.decode("utf-8")
        
        doc_id = file.filename or "uploaded_doc"
        
        rag_instance.add_document(
            doc_id=doc_id,
            content=content_str,
            title=file.filename,
        )
        
        return {
            "status": "uploaded",
            "id": doc_id,
            "size": len(content_str),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """List all documents."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    return {
        "documents": [
            {
                "id": doc_id,
                "title": doc["metadata"].get("title", doc_id),
                "length": len(doc["content"]),
            }
            for doc_id, doc in rag_instance.documents.items()
        ]
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    if doc_id not in rag_instance.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    rag_instance.search.delete_document(doc_id)
    del rag_instance.documents[doc_id]
    
    return {"status": "deleted", "id": doc_id}


# =============================================================================
# Search & Query
# =============================================================================

@app.post("/api/query")
async def query(req: QueryRequest):
    """Query the RAG system."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        response = rag_instance.query(
            question=req.question,
            k=req.k,
            include_graph=req.include_graph,
        )
        
        return {
            "answer": response.answer,
            "sources": [
                {
                    "id": s.id,
                    "content": s.content[:500],
                    "score": s.final_score,
                    "vector_score": s.vector_score,
                    "sparse_score": s.sparse_score,
                    "graph_score": s.graph_score,
                    "methods": s.sources,
                }
                for s in response.sources
            ],
            "stats": response.search_results.stats,
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
            for chunk in rag_instance.query_stream(
                question=req.question,
                k=req.k,
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@app.post("/api/search/compare")
async def compare_search(req: QueryRequest):
    """Compare different search methods."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    comparison = rag_instance.compare_search_methods(
        question=req.question,
        k=req.k,
    )
    
    return comparison


@app.post("/api/search/vector")
async def search_vector(req: QueryRequest):
    """Search using only vector similarity."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    results = rag_instance.search.search_vector_only(req.question, k=req.k)
    
    return {
        "results": [
            {
                "id": r.id,
                "content": r.content[:500],
                "score": r.score,
            }
            for r in results
        ]
    }


@app.post("/api/search/sparse")
async def search_sparse(req: QueryRequest):
    """Search using only BM25."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    results = rag_instance.search.search_sparse_only(req.question, k=req.k)
    
    return {
        "results": [
            {
                "id": r.id,
                "content": r.content[:500],
                "score": r.score,
                "matched_terms": r.matched_terms,
            }
            for r in results
        ]
    }


# =============================================================================
# Knowledge Graph
# =============================================================================

@app.get("/api/graph")
async def get_graph():
    """Get the knowledge graph for visualization."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    graph_data = rag_instance.export_graph()
    
    # Format for visualization (e.g., vis.js)
    return {
        "nodes": [
            {
                "id": e["id"],
                "label": e["name"],
                "group": e["type"],
            }
            for e in graph_data["entities"]
        ],
        "edges": [
            {
                "from": r["source_id"],
                "to": r["target_id"],
                "label": r["type"],
            }
            for r in graph_data["relations"]
        ],
        "stats": rag_instance.search.knowledge_graph.get_stats(),
    }


@app.post("/api/graph/entities")
async def add_entity(entity: EntityRequest):
    """Add an entity to the knowledge graph."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    rag_instance.add_entity(
        entity_id=entity.id,
        name=entity.name,
        entity_type=entity.type,
        properties=entity.properties,
    )
    
    return {"status": "added", "id": entity.id}


@app.post("/api/graph/relations")
async def add_relation(relation: RelationRequest):
    """Add a relation to the knowledge graph."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    rag_instance.add_relation(
        source_id=relation.source_id,
        target_id=relation.target_id,
        relation_type=relation.type,
        weight=relation.weight,
    )
    
    return {"status": "added"}


@app.get("/api/graph/entities/{entity_id}/related")
async def get_related_entities(entity_id: str, max_depth: int = 2):
    """Get entities related to a given entity."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    related = rag_instance.get_related_entities(entity_id, max_depth)
    
    return {"entity_id": entity_id, "related": related}


# =============================================================================
# Run server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
