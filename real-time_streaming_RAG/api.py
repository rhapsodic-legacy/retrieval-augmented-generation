"""
FastAPI Backend for Real-Time Streaming RAG

Features:
- REST API for queries and data ingestion
- WebSocket for real-time stream updates
- Server-Sent Events for streaming responses
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from streaming_rag import StreamingRAG, StreamingRAGConfig


app = FastAPI(
    title="Streaming RAG API",
    description="Real-time RAG over streaming data with freshness awareness",
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
rag_instance: Optional[StreamingRAG] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()


# =============================================================================
# Pydantic Models
# =============================================================================

class ConfigRequest(BaseModel):
    llm_provider: str = "anthropic"
    llm_model: Optional[str] = None
    embedding_provider: str = "local"
    time_decay_hours: float = 24
    n_results: int = 10


class QueryRequest(BaseModel):
    question: str
    k: int = 10
    content_types: Optional[list[str]] = None
    sources: Optional[list[str]] = None
    time_range_hours: Optional[float] = None


class AddItemRequest(BaseModel):
    content: str
    content_type: str = "generic"
    source: str = "api"
    title: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: dict = {}


class WebhookData(BaseModel):
    content: str
    type: str = "generic"
    source: Optional[str] = None
    title: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: dict = {}


# =============================================================================
# Event handlers
# =============================================================================

def on_new_item(item):
    """Broadcast new items to WebSocket clients."""
    asyncio.create_task(manager.broadcast({
        "type": "new_item",
        "data": item.to_dict(),
    }))


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Streaming RAG API</h1><p>Visit /docs for API documentation</p>"


@app.post("/api/init")
async def initialize(config: ConfigRequest):
    """Initialize the Streaming RAG system."""
    global rag_instance
    
    try:
        rag_config = StreamingRAGConfig(
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            embedding_provider=config.embedding_provider,
            time_decay_hours=config.time_decay_hours,
            n_results=config.n_results,
        )
        
        rag_instance = StreamingRAG(rag_config)
        rag_instance.on_new_item = on_new_item
        
        return {"status": "initialized", "config": config.dict()}
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
        "sources": rag_instance.get_sources(),
    }


# =============================================================================
# Streaming Control
# =============================================================================

@app.post("/api/streams/demo/start")
async def start_demo_streams(
    news_rate: float = 3,
    social_rate: float = 8,
):
    """Start demo data streams."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    rag_instance.start_demo_streams(news_rate, social_rate)
    return {"status": "started", "streams": ["demo-news", "demo-social"]}


@app.post("/api/streams/stop")
async def stop_streams():
    """Stop all data streams."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    rag_instance.stop_streams()
    return {"status": "stopped"}


@app.post("/api/streams/rss")
async def add_rss_feed(url: str, name: Optional[str] = None):
    """Add an RSS feed source."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    rag_instance.add_rss_feed(url, name)
    return {"status": "added", "url": url, "name": name}


# =============================================================================
# Data Ingestion
# =============================================================================

@app.post("/api/items")
async def add_item(req: AddItemRequest):
    """Add a single item to the stream."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    timestamp = None
    if req.timestamp:
        try:
            timestamp = datetime.fromisoformat(req.timestamp)
        except ValueError:
            pass
    
    result = rag_instance.add_item(
        content=req.content,
        content_type=req.content_type,
        source=req.source,
        title=req.title,
        timestamp=timestamp,
        **req.metadata,
    )
    
    return result


@app.post("/api/webhook")
async def webhook(data: WebhookData):
    """Receive data via webhook."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    result = rag_instance.ingest_webhook(data.dict())
    
    if result:
        return {"status": "ingested", "item": result}
    return {"status": "rejected"}


# =============================================================================
# Query
# =============================================================================

@app.post("/api/query")
async def query(req: QueryRequest):
    """Query the streaming data."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        response = rag_instance.query(
            question=req.question,
            k=req.k,
            content_types=req.content_types,
            sources=req.sources,
            time_range_hours=req.time_range_hours,
        )
        
        return {
            "answer": response.answer,
            "freshness_note": response.freshness_note,
            "as_of": response.as_of.isoformat(),
            "sources": response.context.sources,
            "results": [
                {
                    "id": r.item.id,
                    "content": r.item.content[:300],
                    "title": r.item.title,
                    "type": r.item.content_type.value,
                    "source": r.item.source,
                    "freshness": r.item.freshness_label,
                    "timestamp": r.item.timestamp.isoformat(),
                    "scores": {
                        "semantic": round(r.semantic_score, 3),
                        "time": round(r.time_score, 3),
                        "priority": round(r.priority_score, 3),
                        "final": round(r.final_score, 3),
                    },
                }
                for r in response.context.results
            ],
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
# Data Access
# =============================================================================

@app.get("/api/recent")
async def get_recent(
    n: int = 20,
    content_types: Optional[str] = None,
):
    """Get most recent items."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    types = content_types.split(",") if content_types else None
    items = rag_instance.get_recent(n, types)
    
    return {"items": items, "count": len(items)}


@app.get("/api/sources")
async def get_sources():
    """Get list of data sources."""
    if not rag_instance:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    return {"sources": rag_instance.get_sources()}


# =============================================================================
# WebSocket
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial status
        if rag_instance:
            await websocket.send_json({
                "type": "status",
                "data": rag_instance.get_stats(),
            })
        
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle commands
            try:
                msg = json.loads(data)
                
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif msg.get("type") == "get_recent" and rag_instance:
                    items = rag_instance.get_recent(msg.get("n", 10))
                    await websocket.send_json({
                        "type": "recent",
                        "data": items,
                    })
                    
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
