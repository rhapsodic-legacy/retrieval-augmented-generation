"""
Time-Weighted Streaming Index

Features:
- Incremental updates without full rebuild
- Time-decay scoring (recent = more relevant)
- Automatic cleanup of old content
- Priority-aware retrieval
"""

from dataclasses import dataclass
from typing import Optional, Callable
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import math
import threading
import time

from ..ingestion.content_types import StreamItem, ContentType, ContentPriority
from ..providers.embeddings import BaseEmbedding
from ..dedup import DeduplicationEngine, DedupConfig


@dataclass
class IndexConfig:
    """Configuration for the streaming index."""
    
    # Time decay
    decay_function: str = "exponential"  # "exponential", "linear", "logarithmic"
    half_life_hours: float = 24  # Time for relevance to decay by 50%
    
    # Freshness boost
    freshness_boost_hours: float = 1  # Extra boost for very recent content
    freshness_boost_factor: float = 1.5
    
    # Cleanup
    max_age_days: int = 30
    cleanup_interval_hours: float = 1
    
    # Deduplication
    enable_dedup: bool = True
    dedup_threshold: float = 0.8
    
    # Index limits
    max_items: int = 100000


@dataclass
class SearchResult:
    """A search result with scores."""
    item: StreamItem
    semantic_score: float
    time_score: float
    priority_score: float
    final_score: float
    
    @property
    def freshness(self) -> str:
        return self.item.freshness_label


class StreamingIndex:
    """
    Time-weighted index for streaming content.
    
    Features:
    - Incremental updates (add/remove without rebuild)
    - Time-decay scoring
    - Priority weighting
    - Automatic cleanup
    - Deduplication
    """
    
    def __init__(
        self,
        embedding_provider: BaseEmbedding,
        config: Optional[IndexConfig] = None,
    ):
        self.config = config or IndexConfig()
        self.embedding_provider = embedding_provider
        
        # Storage
        self.items: dict[str, StreamItem] = {}
        
        # Indexes
        self.by_source: dict[str, set[str]] = defaultdict(set)
        self.by_type: dict[ContentType, set[str]] = defaultdict(set)
        self.by_time: list[tuple[datetime, str]] = []  # Sorted by timestamp
        
        # Vector store
        self._vector_store = None
        self._init_vector_store()
        
        # Deduplication
        self.dedup_engine = None
        if self.config.enable_dedup:
            self.dedup_engine = DeduplicationEngine(DedupConfig(
                similarity_threshold=self.config.dedup_threshold,
            ))
        
        # Cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Statistics
        self.stats = {
            "items_added": 0,
            "items_removed": 0,
            "duplicates_detected": 0,
            "searches": 0,
        }
    
    def _init_vector_store(self):
        """Initialize ChromaDB for vector search."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
            
            self._vector_store = self._client.get_or_create_collection(
                name="streaming_content",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"Warning: Failed to initialize vector store: {e}")
            self._vector_store = None
    
    # =========================================================================
    # Incremental Updates
    # =========================================================================
    
    def add(self, item: StreamItem) -> dict:
        """
        Add an item to the index incrementally.
        
        Returns status dict with duplicate info.
        """
        result = {
            "status": "added",
            "id": item.id,
            "is_duplicate": False,
        }
        
        # Check for duplicates
        if self.dedup_engine:
            is_dup, original_id, similarity = self.dedup_engine.check_and_add(
                item.id,
                item.content,
                item.content_hash,
            )
            
            if is_dup:
                item.is_duplicate = True
                item.duplicate_of = original_id
                result["is_duplicate"] = True
                result["duplicate_of"] = original_id
                result["similarity"] = similarity
                self.stats["duplicates_detected"] += 1
        
        # Generate embedding
        if item.embedding is None:
            item.embedding = self.embedding_provider.embed(item.content)
        
        # Generate fingerprint for dedup
        if self.dedup_engine and item.fingerprint is None:
            item.fingerprint = self.dedup_engine.get_signature(item.content)
        
        # Store item
        self.items[item.id] = item
        
        # Update indexes
        self.by_source[item.source].add(item.id)
        self.by_type[item.content_type].add(item.id)
        
        # Insert into time-sorted list (maintain order)
        self._insert_by_time(item)
        
        # Add to vector store
        if self._vector_store and item.embedding:
            self._vector_store.upsert(
                ids=[item.id],
                embeddings=[item.embedding],
                documents=[item.content],
                metadatas=[{
                    "source": item.source,
                    "content_type": item.content_type.value,
                    "timestamp": item.timestamp.isoformat(),
                    "priority": item.priority.value,
                    "is_duplicate": item.is_duplicate,
                }],
            )
        
        self.stats["items_added"] += 1
        
        # Check if cleanup needed
        if len(self.items) > self.config.max_items:
            self._cleanup_oldest(len(self.items) - self.config.max_items)
        
        return result
    
    def add_batch(self, items: list[StreamItem]) -> list[dict]:
        """Add multiple items."""
        results = []
        
        # Batch embed
        texts = [item.content for item in items if item.embedding is None]
        if texts:
            embeddings = self.embedding_provider.embed_batch(texts)
            embed_idx = 0
            for item in items:
                if item.embedding is None:
                    item.embedding = embeddings[embed_idx]
                    embed_idx += 1
        
        # Add each item
        for item in items:
            results.append(self.add(item))
        
        return results
    
    def remove(self, item_id: str) -> bool:
        """Remove an item from the index."""
        if item_id not in self.items:
            return False
        
        item = self.items[item_id]
        
        # Remove from indexes
        self.by_source[item.source].discard(item_id)
        self.by_type[item.content_type].discard(item_id)
        
        # Remove from time list
        self.by_time = [(t, i) for t, i in self.by_time if i != item_id]
        
        # Remove from vector store
        if self._vector_store:
            try:
                self._vector_store.delete(ids=[item_id])
            except Exception:
                pass
        
        # Remove from dedup
        if self.dedup_engine:
            self.dedup_engine.remove(item_id)
        
        del self.items[item_id]
        self.stats["items_removed"] += 1
        
        return True
    
    def _insert_by_time(self, item: StreamItem):
        """Insert item into time-sorted list."""
        # Binary search for insertion point
        import bisect
        
        # Convert to tuple for comparison
        entry = (item.timestamp, item.id)
        
        # Find insertion point (newest first, so reverse comparison)
        timestamps = [t for t, _ in self.by_time]
        idx = bisect.bisect_left(timestamps, item.timestamp)
        
        self.by_time.insert(idx, entry)
    
    # =========================================================================
    # Time-Weighted Search
    # =========================================================================
    
    def search(
        self,
        query: str,
        k: int = 10,
        content_types: Optional[list[ContentType]] = None,
        sources: Optional[list[str]] = None,
        min_timestamp: Optional[datetime] = None,
        max_timestamp: Optional[datetime] = None,
        include_duplicates: bool = False,
        time_weight: float = 0.3,
        priority_weight: float = 0.1,
    ) -> list[SearchResult]:
        """
        Search with time-weighted ranking.
        
        Final score = semantic_score * (1 - time_weight - priority_weight)
                    + time_score * time_weight
                    + priority_score * priority_weight
        """
        self.stats["searches"] += 1
        
        if not self._vector_store:
            return self._fallback_search(query, k, content_types, include_duplicates)
        
        # Build filter
        where = {}
        if not include_duplicates:
            where["is_duplicate"] = False
        if content_types:
            where["content_type"] = {"$in": [t.value for t in content_types]}
        
        # Embed query
        query_embedding = self.embedding_provider.embed(query)
        
        # Search vector store (over-fetch for filtering)
        results = self._vector_store.query(
            query_embeddings=[query_embedding],
            n_results=k * 3,
            where=where if where else None,
            include=["distances", "metadatas"],
        )
        
        # Process results with time weighting
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, item_id in enumerate(results["ids"][0]):
                item = self.items.get(item_id)
                if not item:
                    continue
                
                # Apply filters
                if sources and item.source not in sources:
                    continue
                if min_timestamp and item.timestamp < min_timestamp:
                    continue
                if max_timestamp and item.timestamp > max_timestamp:
                    continue
                
                # Calculate scores
                distance = results["distances"][0][i] if results["distances"] else 0
                semantic_score = 1 - distance
                
                time_score = self._calculate_time_score(item.timestamp)
                priority_score = item.priority.value / 5.0
                
                # Combine scores
                semantic_weight = 1 - time_weight - priority_weight
                final_score = (
                    semantic_score * semantic_weight +
                    time_score * time_weight +
                    priority_score * priority_weight
                )
                
                search_results.append(SearchResult(
                    item=item,
                    semantic_score=semantic_score,
                    time_score=time_score,
                    priority_score=priority_score,
                    final_score=final_score,
                ))
        
        # Sort by final score
        search_results.sort(key=lambda r: r.final_score, reverse=True)
        
        return search_results[:k]
    
    def _calculate_time_score(self, timestamp: datetime) -> float:
        """Calculate time decay score."""
        now = datetime.now(timezone.utc)
        
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        age_hours = (now - timestamp).total_seconds() / 3600
        
        # Apply freshness boost for very recent content
        if age_hours <= self.config.freshness_boost_hours:
            boost = self.config.freshness_boost_factor
        else:
            boost = 1.0
        
        # Calculate decay
        if self.config.decay_function == "exponential":
            # Exponential decay: score = e^(-λt) where λ = ln(2)/half_life
            decay_rate = math.log(2) / self.config.half_life_hours
            score = math.exp(-decay_rate * age_hours)
        
        elif self.config.decay_function == "linear":
            # Linear decay over max_age_days
            max_hours = self.config.max_age_days * 24
            score = max(0, 1 - (age_hours / max_hours))
        
        elif self.config.decay_function == "logarithmic":
            # Logarithmic decay (slower decline)
            score = 1 / (1 + math.log1p(age_hours / self.config.half_life_hours))
        
        else:
            score = 1.0
        
        return min(score * boost, 1.0)
    
    def _fallback_search(
        self,
        query: str,
        k: int,
        content_types: Optional[list[ContentType]],
        include_duplicates: bool,
    ) -> list[SearchResult]:
        """Fallback text search when vector store unavailable."""
        query_lower = query.lower()
        results = []
        
        for item in self.items.values():
            if not include_duplicates and item.is_duplicate:
                continue
            if content_types and item.content_type not in content_types:
                continue
            
            if query_lower in item.content.lower():
                time_score = self._calculate_time_score(item.timestamp)
                
                results.append(SearchResult(
                    item=item,
                    semantic_score=0.5,
                    time_score=time_score,
                    priority_score=item.priority.value / 5.0,
                    final_score=0.5 + time_score * 0.3,
                ))
        
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:k]
    
    # =========================================================================
    # Time-based queries
    # =========================================================================
    
    def get_recent(
        self,
        n: int = 10,
        content_types: Optional[list[ContentType]] = None,
        include_duplicates: bool = False,
    ) -> list[StreamItem]:
        """Get most recent items."""
        results = []
        
        # Iterate from newest (end of sorted list)
        for timestamp, item_id in reversed(self.by_time):
            if len(results) >= n:
                break
            
            item = self.items.get(item_id)
            if not item:
                continue
            
            if not include_duplicates and item.is_duplicate:
                continue
            if content_types and item.content_type not in content_types:
                continue
            
            results.append(item)
        
        return results
    
    def get_by_time_range(
        self,
        start: datetime,
        end: datetime,
        content_types: Optional[list[ContentType]] = None,
    ) -> list[StreamItem]:
        """Get items in a time range."""
        results = []
        
        for timestamp, item_id in self.by_time:
            if timestamp < start:
                continue
            if timestamp > end:
                break
            
            item = self.items.get(item_id)
            if not item:
                continue
            
            if content_types and item.content_type not in content_types:
                continue
            
            results.append(item)
        
        return results
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        """Stop background cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        interval = self.config.cleanup_interval_hours * 3600
        
        while not self._stop_cleanup.is_set():
            self._cleanup_old()
            self._stop_cleanup.wait(interval)
    
    def _cleanup_old(self):
        """Remove items older than max_age_days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.max_age_days)
        
        to_remove = []
        for timestamp, item_id in self.by_time:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            if timestamp < cutoff:
                to_remove.append(item_id)
            else:
                break  # List is sorted, no more old items
        
        for item_id in to_remove:
            self.remove(item_id)
    
    def _cleanup_oldest(self, n: int):
        """Remove n oldest items."""
        to_remove = []
        
        for _, item_id in self.by_time[:n]:
            to_remove.append(item_id)
        
        for item_id in to_remove:
            self.remove(item_id)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        type_counts = {t.value: len(ids) for t, ids in self.by_type.items()}
        source_counts = {s: len(ids) for s, ids in self.by_source.items()}
        
        return {
            "total_items": len(self.items),
            "by_type": type_counts,
            "by_source": source_counts,
            "operations": self.stats,
            "dedup": self.dedup_engine.get_stats() if self.dedup_engine else None,
        }
    
    def get_sources(self) -> list[str]:
        """Get list of sources."""
        return list(self.by_source.keys())
