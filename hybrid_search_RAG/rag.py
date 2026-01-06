"""
Hybrid Search RAG

Production-grade RAG combining:
- Vector search (semantic)
- BM25 search (keyword)
- Knowledge graph (relationships)
- Learned fusion (ML re-ranking)

With multi-provider support for LLMs and embeddings.
"""

from dataclasses import dataclass, field
from typing import Optional, Generator
from pathlib import Path

from .providers import create_llm, create_embedding, BaseLLM, BaseEmbedding
from .search import (
    HybridSearch,
    HybridSearchConfig,
    HybridSearchResult,
    KnowledgeGraph,
    Entity,
    EntityType,
    Relation,
    RelationType,
)
from .fusion import create_fusion, FusedResult


@dataclass
class RAGConfig:
    """Configuration for Hybrid RAG."""
    
    # LLM settings
    llm_provider: str = "anthropic"  # "anthropic", "gemini", "openai"
    llm_model: Optional[str] = None
    
    # Embedding settings
    embedding_provider: str = "local"  # "voyage", "openai", "google", "sentence_transformers", "local"
    embedding_model: Optional[str] = None
    
    # Search settings
    enable_vector: bool = True
    enable_sparse: bool = True
    enable_graph: bool = True
    
    # Fusion settings
    fusion_strategy: str = "rrf"  # "rrf", "weighted", "learned"
    vector_weight: float = 0.4
    sparse_weight: float = 0.35
    graph_weight: float = 0.25
    
    # Retrieval settings
    n_results: int = 5
    
    # Generation settings
    max_tokens: int = 1000
    temperature: float = 0.1
    
    # Persistence
    persist_directory: Optional[str] = None


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    sources: list[FusedResult]
    
    # Search breakdown
    search_results: HybridSearchResult
    
    # Metadata
    query: str
    model: str
    metadata: dict = field(default_factory=dict)


class HybridRAG:
    """
    Hybrid Search RAG with multi-provider support.
    
    Combines:
    - Dense retrieval (vector embeddings)
    - Sparse retrieval (BM25)
    - Graph retrieval (knowledge graph)
    - ML fusion (learned re-ranking)
    
    Usage:
        rag = HybridRAG()
        rag.add_document("doc1", "Machine learning is...")
        
        response = rag.query("What is machine learning?")
        print(response.answer)
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
- Answer based on the context provided
- If the context doesn't contain the answer, say so
- Be concise but complete
- Cite sources when appropriate using [Source: title] format
- If multiple sources agree, synthesize them"""

    QUERY_PROMPT = """Based on the following context, answer the user's question.

CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive answer based on the context. Cite relevant sources."""

    ENTITY_EXTRACTION_PROMPT = """Extract key entities from this text. Return as JSON array.

TEXT:
{text}

Return entities in this format:
[
  {{"name": "Entity Name", "type": "person|organization|location|concept|product|event|other"}}
]

Only return the JSON array, nothing else."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize Hybrid RAG."""
        self.config = config or RAGConfig()
        
        # Initialize providers
        self.llm = create_llm(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )
        
        self.embeddings = create_embedding(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
        )
        
        # Initialize search
        search_config = HybridSearchConfig(
            enable_vector=self.config.enable_vector,
            enable_sparse=self.config.enable_sparse,
            enable_graph=self.config.enable_graph,
            vector_weight=self.config.vector_weight,
            sparse_weight=self.config.sparse_weight,
            graph_weight=self.config.graph_weight,
            fusion_strategy=self.config.fusion_strategy,
            final_k=self.config.n_results,
        )
        
        self.search = HybridSearch(
            embedding_provider=self.embeddings,
            config=search_config,
        )
        
        # Document tracking
        self.documents: dict[str, dict] = {}
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
        extract_entities: bool = True,
    ):
        """
        Add a document to the RAG system.
        
        Args:
            doc_id: Unique document ID
            content: Document text
            title: Optional title
            metadata: Optional metadata
            extract_entities: Whether to extract entities for knowledge graph
        """
        metadata = metadata or {}
        metadata["title"] = title or doc_id
        
        # Track document
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata,
        }
        
        # Extract entities if enabled
        entities = []
        relations = []
        
        if extract_entities and self.config.enable_graph:
            try:
                entities, relations = self._extract_entities(doc_id, content)
            except Exception as e:
                print(f"Warning: Entity extraction failed: {e}")
        
        # Add to search index
        self.search.add_document(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            entities=entities,
            relations=relations,
        )
    
    def add_documents(self, documents: list[dict]):
        """
        Add multiple documents.
        
        Each document should have:
        - id: str
        - content: str
        - title: Optional[str]
        - metadata: Optional[dict]
        """
        for doc in documents:
            self.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                title=doc.get("title"),
                metadata=doc.get("metadata"),
            )
    
    def _extract_entities(
        self,
        doc_id: str,
        content: str,
    ) -> tuple[list[dict], list[dict]]:
        """Extract entities from document using LLM."""
        import json
        import re
        
        # Truncate for extraction
        text = content[:2000]
        
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        response = self.llm.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0,
        )
        
        # Parse JSON
        try:
            json_match = re.search(r'\[[\s\S]*\]', response.content)
            if json_match:
                entity_data = json.loads(json_match.group())
            else:
                return [], []
        except json.JSONDecodeError:
            return [], []
        
        entities = []
        relations = []
        
        for i, ent in enumerate(entity_data[:10]):  # Limit entities
            entity_id = f"{doc_id}_entity_{i}"
            
            entities.append({
                "id": entity_id,
                "name": ent.get("name", f"Entity {i}"),
                "type": ent.get("type", "other"),
                "properties": {"source_doc": doc_id},
            })
            
            # Add relation to document
            relations.append({
                "source_id": doc_id,
                "target_id": entity_id,
                "type": "mentions",
            })
        
        return entities, relations
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        include_graph: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG system.
        
        Args:
            question: User question
            k: Number of results (uses config if not specified)
            include_graph: Whether to include graph search
            
        Returns:
            RAGResponse with answer and sources
        """
        k = k or self.config.n_results
        
        # Identify query entities for graph search
        query_entities = None
        if include_graph and self.config.enable_graph:
            query_entities = self._identify_query_entities(question)
        
        # Search
        search_results = self.search.search(
            query=question,
            k=k,
            query_entities=query_entities,
        )
        
        # Build context from results
        context = self._build_context(search_results.results)
        
        # Generate answer
        prompt = self.QUERY_PROMPT.format(
            context=context,
            question=question,
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        return RAGResponse(
            answer=response.content,
            sources=search_results.results,
            search_results=search_results,
            query=question,
            model=response.model,
        )
    
    def query_stream(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> Generator[str, None, RAGResponse]:
        """Stream a query response."""
        k = k or self.config.n_results
        
        # Search
        search_results = self.search.search(query=question, k=k)
        
        # Build context
        context = self._build_context(search_results.results)
        
        # Generate with streaming
        prompt = self.QUERY_PROMPT.format(
            context=context,
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
        
        return RAGResponse(
            answer=full_answer,
            sources=search_results.results,
            search_results=search_results,
            query=question,
            model=self.llm.model if hasattr(self.llm, 'model') else "unknown",
        )
    
    def _identify_query_entities(self, question: str) -> list[str]:
        """Identify entity IDs relevant to the query."""
        # Simple approach: search for matching entity names
        entity_ids = []
        
        question_lower = question.lower()
        
        for entity in self.search.knowledge_graph.entities.values():
            if entity.name.lower() in question_lower:
                entity_ids.append(entity.id)
        
        return entity_ids[:3]  # Limit to 3 entities
    
    def _build_context(self, results: list[FusedResult]) -> str:
        """Build context string from search results."""
        parts = []
        
        for i, result in enumerate(results, 1):
            title = result.metadata.get("title", result.id)
            
            parts.append(f"[Source {i}: {title}]")
            parts.append(result.content)
            
            # Add search method info
            if result.sources:
                parts.append(f"(Found via: {', '.join(result.sources)})")
            
            parts.append("")
        
        return "\n".join(parts)
    
    # =========================================================================
    # Graph operations
    # =========================================================================
    
    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "other",
        properties: Optional[dict] = None,
    ):
        """Add a custom entity to the knowledge graph."""
        entity = Entity(
            id=entity_id,
            name=name,
            type=EntityType(entity_type),
            properties=properties or {},
        )
        self.search.knowledge_graph.add_entity(entity)
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "related_to",
        weight: float = 1.0,
    ):
        """Add a relation to the knowledge graph."""
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            type=RelationType(relation_type),
            weight=weight,
        )
        self.search.knowledge_graph.add_relation(relation)
    
    def get_related_entities(
        self,
        entity_id: str,
        max_depth: int = 2,
    ) -> list[dict]:
        """Get entities related to a given entity."""
        results = self.search.knowledge_graph.search_by_entity(
            entity_id,
            max_depth=max_depth,
        )
        
        return [
            {
                "id": r.entity.id,
                "name": r.entity.name,
                "type": r.entity.type.value,
                "score": r.score,
                "path": r.path,
            }
            for r in results
        ]
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    def compare_search_methods(
        self,
        question: str,
        k: int = 10,
    ) -> dict:
        """
        Compare results from different search methods.
        
        Useful for debugging and analysis.
        """
        # Vector search
        vector_results = self.search.search_vector_only(question, k)
        
        # Sparse search
        sparse_results = self.search.search_sparse_only(question, k)
        
        # Hybrid search
        hybrid_results = self.search.search(question, k)
        
        # Find unique and overlapping results
        vector_ids = {r.id for r in vector_results}
        sparse_ids = {r.id for r in sparse_results}
        hybrid_ids = {r.id for r in hybrid_results.results}
        
        return {
            "vector": {
                "count": len(vector_results),
                "ids": list(vector_ids),
                "unique": list(vector_ids - sparse_ids),
            },
            "sparse": {
                "count": len(sparse_results),
                "ids": list(sparse_ids),
                "unique": list(sparse_ids - vector_ids),
            },
            "overlap": list(vector_ids & sparse_ids),
            "hybrid": {
                "count": len(hybrid_results.results),
                "ids": list(hybrid_ids),
            },
        }
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "documents": len(self.documents),
            "search": self.search.get_stats(),
            "config": {
                "llm_provider": self.config.llm_provider,
                "embedding_provider": self.config.embedding_provider,
                "fusion_strategy": self.config.fusion_strategy,
            },
        }
    
    def export_graph(self) -> dict:
        """Export knowledge graph for visualization."""
        return self.search.knowledge_graph.to_dict()
