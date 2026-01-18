"""
Hybrid Search Orchestrator 

Combines:
- Vector search (semantic similarity)
- Sparse search (BM25 keyword matching)
- Knowledge graph queries (relationships)

With configurable fusion strategies.
"""

from dataclasses import dataclass, field
from typing import Optional

from .vector import BaseVectorStore, VectorSearchResult, create_vector_store
from .sparse import BM25Index, SparseSearchResult, create_sparse_index
from .knowledge_graph import KnowledgeGraph, GraphSearchResult, Entity, EntityType
from ..fusion.strategies import BaseFusion, FusedResult, create_fusion
from ..providers.embeddings import BaseEmbedding


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    
    # Vector search
    enable_vector: bool = True
    vector_weight: float = 0.4
    vector_k: int = 20  # Candidates to retrieve
    
    # Sparse search
    enable_sparse: bool = True
    sparse_weight: float = 0.35
    sparse_k: int = 20
    
    # Graph search
    enable_graph: bool = True
    graph_weight: float = 0.25
    graph_k: int = 10
    graph_max_depth: int = 2
    
    # Fusion
    fusion_strategy: str = "rrf"  # "rrf", "weighted", "learned"
    final_k: int = 10


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    results: list[FusedResult]
    
    # Individual results (for analysis)
    vector_results: list[VectorSearchResult] = field(default_factory=list)
    sparse_results: list[SparseSearchResult] = field(default_factory=list)
    graph_results: list[GraphSearchResult] = field(default_factory=list)
    
    # Query info
    query: str = ""
    query_entities: list[str] = field(default_factory=list)
    
    # Stats
    stats: dict = field(default_factory=dict)


class HybridSearch:
    """
    Hybrid search combining vector, sparse, and graph retrieval.
    
    Usage:
        search = HybridSearch(embedding_provider)
        
        # Add documents
        search.add_documents([...])
        
        # Search
        results = search.search("What is machine learning?")
    """
    
    def __init__(
        self,
        embedding_provider: BaseEmbedding,
        config: Optional[HybridSearchConfig] = None,
        vector_store: Optional[BaseVectorStore] = None,
        sparse_index: Optional[BM25Index] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        fusion: Optional[BaseFusion] = None,
    ):
        """
        Initialize hybrid search.
        
        Args:
            embedding_provider: Provider for generating embeddings
            config: Search configuration
            vector_store: Optional pre-initialized vector store
            sparse_index: Optional pre-initialized BM25 index
            knowledge_graph: Optional pre-initialized knowledge graph
            fusion: Optional fusion strategy
        """
        self.config = config or HybridSearchConfig()
        self.embeddings = embedding_provider
        
        # Initialize components
        self.vector_store = vector_store or create_vector_store("chroma")
        self.sparse_index = sparse_index or create_sparse_index("bm25")
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        
        # Initialize fusion
        if fusion:
            self.fusion = fusion
        else:
            self.fusion = create_fusion(
                self.config.fusion_strategy,
                vector_weight=self.config.vector_weight,
                sparse_weight=self.config.sparse_weight,
                graph_weight=self.config.graph_weight,
            )
        
        # Document storage for content retrieval
        self.documents: dict[str, dict] = {}
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict] = None,
        entities: Optional[list[dict]] = None,
        relations: Optional[list[dict]] = None,
    ):
        """
        Add a document to all indexes.
        
        Args:
            doc_id: Unique document ID
            content: Document text
            metadata: Optional metadata
            entities: Optional entities to add to graph
            relations: Optional relations to add to graph
        """
        metadata = metadata or {}
        
        # Store document
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata,
        }
        
        # Add to vector store
        if self.config.enable_vector:
            embedding = self.embeddings.embed(content)
            self.vector_store.add(
                ids=[doc_id],
                embeddings=[embedding],
                contents=[content],
                metadatas=[metadata],
            )
        
        # Add to sparse index
        if self.config.enable_sparse:
            self.sparse_index.add(
                ids=[doc_id],
                contents=[content],
                metadatas=[metadata],
            )
        
        # Add to knowledge graph
        if self.config.enable_graph:
            # Add document as entity
            doc_entity = Entity(
                id=doc_id,
                name=metadata.get("title", doc_id),
                type=EntityType.DOCUMENT,
                properties=metadata,
                content=content,
            )
            self.knowledge_graph.add_entity(doc_entity)
            
            # Add custom entities
            if entities:
                for ent in entities:
                    entity = Entity(
                        id=ent["id"],
                        name=ent["name"],
                        type=EntityType(ent.get("type", "other")),
                        properties=ent.get("properties", {}),
                    )
                    self.knowledge_graph.add_entity(entity)
            
            # Add relations
            if relations:
                from .knowledge_graph import Relation, RelationType
                
                for rel in relations:
                    relation = Relation(
                        source_id=rel["source_id"],
                        target_id=rel["target_id"],
                        type=RelationType(rel.get("type", "related_to")),
                        weight=rel.get("weight", 1.0),
                        properties=rel.get("properties", {}),
                    )
                    self.knowledge_graph.add_relation(relation)
    
    def add_documents(
        self,
        documents: list[dict],
    ):
        """
        Add multiple documents.
        
        Each document should have:
        - id: str
        - content: str
        - metadata: Optional[dict]
        - entities: Optional[list]
        - relations: Optional[list]
        """
        for doc in documents:
            self.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata"),
                entities=doc.get("entities"),
                relations=doc.get("relations"),
            )
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
        query_entities: Optional[list[str]] = None,
    ) -> HybridSearchResult:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            k: Number of results (uses config if not specified)
            filter_metadata: Optional metadata filter
            query_entities: Optional entity IDs to use for graph search
            
        Returns:
            HybridSearchResult with fused and individual results
        """
        k = k or self.config.final_k
        
        vector_results = []
        sparse_results = []
        graph_results = []
        
        # Vector search
        if self.config.enable_vector:
            query_embedding = self.embeddings.embed(query)
            vector_results = self.vector_store.search(
                query_embedding,
                k=self.config.vector_k,
                filter_metadata=filter_metadata,
            )
        
        # Sparse search
        if self.config.enable_sparse:
            sparse_results = self.sparse_index.search(
                query,
                k=self.config.sparse_k,
                filter_metadata=filter_metadata,
            )
        
        # Graph search
        if self.config.enable_graph and query_entities:
            all_graph_results = []
            for entity_id in query_entities:
                entity_results = self.knowledge_graph.search_by_entity(
                    entity_id,
                    max_depth=self.config.graph_max_depth,
                    max_results=self.config.graph_k,
                )
                all_graph_results.extend(entity_results)
            
            # Deduplicate and sort
            seen = set()
            graph_results = []
            for r in sorted(all_graph_results, key=lambda x: x.score, reverse=True):
                if r.entity.id not in seen:
                    seen.add(r.entity.id)
                    graph_results.append(r)
        
        # Convert to common format for fusion
        vector_dicts = [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in vector_results
        ]
        
        sparse_dicts = [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in sparse_results
        ]
        
        graph_dicts = [
            {
                "id": r.entity.id,
                "content": r.entity.content or r.entity.name,
                "score": r.score,
                "metadata": r.entity.properties,
            }
            for r in graph_results
        ]
        
        # Fuse results
        fused_results = self.fusion.fuse(
            vector_dicts,
            sparse_dicts,
            graph_dicts,
            k=k,
        )
        
        return HybridSearchResult(
            results=fused_results,
            vector_results=vector_results,
            sparse_results=sparse_results,
            graph_results=graph_results,
            query=query,
            query_entities=query_entities or [],
            stats={
                "vector_count": len(vector_results),
                "sparse_count": len(sparse_results),
                "graph_count": len(graph_results),
                "fused_count": len(fused_results),
            },
        )
    
    def search_vector_only(
        self,
        query: str,
        k: int = 10,
    ) -> list[VectorSearchResult]:
        """Search using only vector similarity."""
        query_embedding = self.embeddings.embed(query)
        return self.vector_store.search(query_embedding, k=k)
    
    def search_sparse_only(
        self,
        query: str,
        k: int = 10,
    ) -> list[SparseSearchResult]:
        """Search using only BM25."""
        return self.sparse_index.search(query, k=k)
    
    def search_graph_only(
        self,
        entity_id: str,
        max_depth: int = 2,
        max_results: int = 10,
    ) -> list[GraphSearchResult]:
        """Search using only knowledge graph."""
        return self.knowledge_graph.search_by_entity(
            entity_id,
            max_depth=max_depth,
            max_results=max_results,
        )
    
    def delete_document(self, doc_id: str):
        """Delete a document from all indexes."""
        self.documents.pop(doc_id, None)
        
        if self.config.enable_vector:
            self.vector_store.delete([doc_id])
        
        if self.config.enable_sparse:
            self.sparse_index.delete([doc_id])
        
        if self.config.enable_graph:
            self.knowledge_graph.delete_entity(doc_id)
    
    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            "documents": len(self.documents),
            "vector_store": self.vector_store.count() if self.config.enable_vector else 0,
            "sparse_index": self.sparse_index.count() if self.config.enable_sparse else 0,
            "knowledge_graph": self.knowledge_graph.get_stats() if self.config.enable_graph else {},
            "config": {
                "enable_vector": self.config.enable_vector,
                "enable_sparse": self.config.enable_sparse,
                "enable_graph": self.config.enable_graph,
                "fusion_strategy": self.config.fusion_strategy,
            },
        }
