"""
Real-Time Streaming RAG

RAG over continuously updating data with:
- Incremental index updates
- Time-weighted retrieval
- Deduplication
- Freshness-aware answers
"""

from dataclasses import dataclass, field
from typing import Optional, Generator, Callable
from datetime import datetime, timezone, timedelta
import threading

from .providers import create_llm, create_embedding, BaseLLM, BaseEmbedding
from .ingestion import (
    StreamItem,
    ContentType,
    ContentPriority,
    create_stream_item,
    BaseIngestor,
    RSSIngestor,
    WebhookIngestor,
    SimulatedNewsIngestor,
    SimulatedSocialIngestor,
)
from .indexing import StreamingIndex, IndexConfig, SearchResult
from .dedup import DedupConfig


@dataclass
class StreamingRAGConfig:
    """
    Configuration for Streaming RAG.
    
    LLM Providers:
        - "anthropic": Claude (claude-sonnet-4-20250514)
        - "gemini": Google Gemini (gemini-1.5-flash) 
        - "openai": OpenAI GPT (gpt-4o-mini)
    
    Embedding Providers:
        - "local": Local embeddings (no API key needed)
        - "voyage": Voyage AI (voyage-3-lite)
        - "openai": OpenAI embeddings
    """
    
    # LLM settings - choose your provider
    llm_provider: str = "gemini"  # Options: "anthropic", "gemini", "openai"
    llm_model: Optional[str] = None  # Uses provider default if None
    
    # Embedding settings
    embedding_provider: str = "local"
    embedding_model: Optional[str] = None
    
    # Index settings
    time_decay_hours: float = 24
    max_age_days: int = 7
    
    # Deduplication
    enable_dedup: bool = True
    dedup_threshold: float = 0.8
    
    # Search settings
    n_results: int = 10
    time_weight: float = 0.3
    priority_weight: float = 0.1
    
    # Generation
    max_tokens: int = 2000
    temperature: float = 0.1


@dataclass
class StreamingContext:
    """Context retrieved for a query."""
    results: list[SearchResult]
    query_time: datetime
    oldest_result: Optional[datetime] = None
    newest_result: Optional[datetime] = None
    sources: list[str] = field(default_factory=list)
    
    @property
    def freshness_summary(self) -> str:
        """Summarize content freshness."""
        if not self.results:
            return "No results found"
        
        if self.newest_result:
            newest = self.results[0].item if self.results else None
            if newest:
                return f"Most recent: {newest.freshness_label}"
        
        return f"{len(self.results)} results found"


@dataclass
class StreamingResponse:
    """Response from Streaming RAG."""
    answer: str
    context: StreamingContext
    query: str
    model: str
    
    # Freshness metadata
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    freshness_note: Optional[str] = None
    
    metadata: dict = field(default_factory=dict)


class StreamingRAG:
    """
    Real-Time Streaming RAG System.
    
    Features:
    - Continuous data ingestion
    - Incremental index updates
    - Time-weighted retrieval
    - Deduplication
    - Freshness-aware answers
    
    Usage:
        rag = StreamingRAG()
        
        # Start simulated data streams
        rag.start_demo_streams()
        
        # Query with freshness awareness
        response = rag.query("What's the latest on AI?")
        print(response.answer)
        print(f"As of: {response.freshness_note}")
        
        # Add custom data
        rag.add_item("Breaking news about...", source="custom")
    """
    
    SYSTEM_PROMPT = """You are a real-time information assistant that answers questions based on streaming data sources including news, social media, and logs.

IMPORTANT GUIDELINES:
1. Always indicate the freshness of information: "As of [time], ..."
2. Prioritize recent information over older data
3. Note if information may be rapidly changing
4. Distinguish between confirmed facts and developing stories
5. Cite sources when possible
6. If data seems outdated, mention that newer information may be available

Format your response to clearly communicate the timeliness of the information."""

    QUERY_PROMPT = """Based on the following real-time data, answer the question.

CURRENT TIME: {current_time}

{context}

QUESTION: {question}

Provide a comprehensive answer that:
1. Starts with freshness context (e.g., "As of 10 minutes ago...")
2. Synthesizes information from multiple sources
3. Notes any conflicting or rapidly changing information
4. Indicates confidence level based on data freshness"""

    def __init__(self, config: Optional[StreamingRAGConfig] = None):
        """Initialize Streaming RAG."""
        self.config = config or StreamingRAGConfig()
        
        # Initialize LLM
        self.llm = create_llm(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )
        
        # Initialize embeddings
        self.embeddings = create_embedding(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
        )
        
        # Initialize index
        index_config = IndexConfig(
            half_life_hours=self.config.time_decay_hours,
            max_age_days=self.config.max_age_days,
            enable_dedup=self.config.enable_dedup,
            dedup_threshold=self.config.dedup_threshold,
        )
        self.index = StreamingIndex(self.embeddings, index_config)
        
        # Ingestors
        self.ingestors: dict[str, BaseIngestor] = {}
        self.webhook_ingestor = WebhookIngestor("webhook")
        self.webhook_ingestor.set_callback(self._on_item)
        self.webhook_ingestor.start()
        
        # Event callbacks
        self.on_new_item: Optional[Callable[[StreamItem], None]] = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    # =========================================================================
    # Data Ingestion
    # =========================================================================
    
    def add_item(
        self,
        content: str,
        content_type: str = "generic",
        source: str = "manual",
        timestamp: Optional[datetime] = None,
        title: Optional[str] = None,
        **metadata,
    ) -> dict:
        """
        Add a single item to the stream.
        
        Returns status including duplicate detection.
        """
        item = create_stream_item(
            content=content,
            content_type=content_type,
            source=source,
            timestamp=timestamp or datetime.now(timezone.utc),
            title=title,
            **metadata,
        )
        
        with self._lock:
            result = self.index.add(item)
        
        # Notify listeners
        if self.on_new_item and not result.get("is_duplicate"):
            self.on_new_item(item)
        
        return result
    
    def add_items(self, items: list[dict]) -> list[dict]:
        """Add multiple items."""
        results = []
        stream_items = []
        
        for item_data in items:
            item = create_stream_item(**item_data)
            stream_items.append(item)
        
        with self._lock:
            results = self.index.add_batch(stream_items)
        
        return results
    
    def ingest_webhook(self, data: dict) -> Optional[dict]:
        """Ingest data from a webhook."""
        item = self.webhook_ingestor.ingest(data)
        if item:
            return item.to_dict()
        return None
    
    def _on_item(self, item: StreamItem):
        """Callback when ingestor emits an item."""
        with self._lock:
            self.index.add(item)
        
        if self.on_new_item and not item.is_duplicate:
            self.on_new_item(item)
    
    # =========================================================================
    # Stream Sources
    # =========================================================================
    
    def add_rss_feed(self, url: str, source_name: Optional[str] = None):
        """Add an RSS feed source."""
        name = source_name or f"rss-{len(self.ingestors)}"
        
        ingestor = RSSIngestor([url], source_name=name)
        ingestor.set_callback(self._on_item)
        ingestor.start()
        
        self.ingestors[name] = ingestor
    
    def start_demo_streams(self, news_rate: float = 3, social_rate: float = 8):
        """Start simulated data streams for demo."""
        # News stream
        news_ingestor = SimulatedNewsIngestor(
            source_name="demo-news",
            items_per_minute=news_rate,
        )
        news_ingestor.set_callback(self._on_item)
        news_ingestor.start()
        self.ingestors["demo-news"] = news_ingestor
        
        # Social stream
        social_ingestor = SimulatedSocialIngestor(
            source_name="demo-social",
            items_per_minute=social_rate,
        )
        social_ingestor.set_callback(self._on_item)
        social_ingestor.start()
        self.ingestors["demo-social"] = social_ingestor
    
    def stop_streams(self):
        """Stop all stream sources."""
        for ingestor in self.ingestors.values():
            ingestor.stop()
        self.ingestors.clear()
    
    # =========================================================================
    # Query
    # =========================================================================
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        content_types: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
        time_range_hours: Optional[float] = None,
    ) -> StreamingResponse:
        """
        Query the streaming data with freshness awareness.
        
        Args:
            question: Natural language question
            k: Number of results
            content_types: Filter by type (news, social, log, etc.)
            sources: Filter by source
            time_range_hours: Only search within this time range
        """
        k = k or self.config.n_results
        query_time = datetime.now(timezone.utc)
        
        # Convert content types
        type_filters = None
        if content_types:
            type_filters = [ContentType(t) for t in content_types]
        
        # Time range filter
        min_timestamp = None
        if time_range_hours:
            min_timestamp = query_time - timedelta(hours=time_range_hours)
        
        # Search
        results = self.index.search(
            query=question,
            k=k,
            content_types=type_filters,
            sources=sources,
            min_timestamp=min_timestamp,
            time_weight=self.config.time_weight,
            priority_weight=self.config.priority_weight,
        )
        
        # Build context
        context = StreamingContext(
            results=results,
            query_time=query_time,
            sources=list(set(r.item.source for r in results)),
        )
        
        if results:
            timestamps = [r.item.timestamp for r in results]
            context.oldest_result = min(timestamps)
            context.newest_result = max(timestamps)
        
        # Build context string
        context_str = self._format_context(results)
        
        # Generate answer
        prompt = self.QUERY_PROMPT.format(
            current_time=query_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            context=context_str,
            question=question,
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        # Build freshness note
        freshness_note = self._build_freshness_note(results, query_time)
        
        return StreamingResponse(
            answer=response.content,
            context=context,
            query=question,
            model=response.model,
            as_of=query_time,
            freshness_note=freshness_note,
        )
    
    def query_stream(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> Generator[str, None, StreamingResponse]:
        """Stream a query response."""
        k = k or self.config.n_results
        query_time = datetime.now(timezone.utc)
        
        results = self.index.search(question, k=k)
        
        context = StreamingContext(
            results=results,
            query_time=query_time,
            sources=list(set(r.item.source for r in results)),
        )
        
        context_str = self._format_context(results)
        
        prompt = self.QUERY_PROMPT.format(
            current_time=query_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            context=context_str,
            question=question,
        )
        
        full_answer = ""
        for chunk in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ):
            full_answer += chunk
            yield chunk
        
        return StreamingResponse(
            answer=full_answer,
            context=context,
            query=question,
            model=getattr(self.llm, 'model', 'unknown'),
            as_of=query_time,
            freshness_note=self._build_freshness_note(results, query_time),
        )
    
    def _format_context(self, results: list[SearchResult]) -> str:
        """Format results for the prompt."""
        if not results:
            return "No relevant data found."
        
        parts = []
        
        for i, result in enumerate(results, 1):
            item = result.item
            
            parts.append(f"[{i}] {item.content_type.value.upper()} | {item.freshness_label}")
            parts.append(f"    Source: {item.source}")
            
            if item.title:
                parts.append(f"    Title: {item.title}")
            
            parts.append(f"    Content: {item.content[:500]}")
            parts.append(f"    Scores: semantic={result.semantic_score:.2f}, time={result.time_score:.2f}")
            parts.append("")
        
        return "\n".join(parts)
    
    def _build_freshness_note(
        self,
        results: list[SearchResult],
        query_time: datetime,
    ) -> str:
        """Build a freshness note for the response."""
        if not results:
            return "No recent data available"
        
        newest = results[0].item
        age_seconds = (query_time - newest.timestamp).total_seconds()
        
        if age_seconds < 60:
            return "Based on data from the last minute"
        elif age_seconds < 3600:
            mins = int(age_seconds / 60)
            return f"As of {mins} minute{'s' if mins != 1 else ''} ago"
        elif age_seconds < 86400:
            hours = int(age_seconds / 3600)
            return f"As of {hours} hour{'s' if hours != 1 else ''} ago"
        else:
            return f"As of {newest.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_recent(
        self,
        n: int = 10,
        content_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """Get most recent items."""
        type_filters = None
        if content_types:
            type_filters = [ContentType(t) for t in content_types]
        
        items = self.index.get_recent(n, type_filters)
        return [item.to_dict() for item in items]
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "index": self.index.get_stats(),
            "active_streams": list(self.ingestors.keys()),
            "config": {
                "llm_provider": self.config.llm_provider,
                "embedding_provider": self.config.embedding_provider,
            },
        }
    
    def get_sources(self) -> list[str]:
        """Get list of data sources."""
        return self.index.get_sources()
    
    def cleanup(self):
        """Stop all streams and cleanup."""
        self.stop_streams()
        self.index.stop_cleanup_thread()
