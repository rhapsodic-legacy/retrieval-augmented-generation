"""
Self-Reflective Retrieval System

Features:
- Retrieves documents from knowledge base
- Evaluates if retrieved docs answer the question
- Iteratively improves retrieval if needed
- Falls back to web search when insufficient
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..providers.llm import BaseLLM
from ..providers.embeddings import BaseEmbedding


class RetrievalQuality(Enum):
    EXCELLENT = "excellent"    # Docs clearly answer the question
    GOOD = "good"              # Docs mostly answer, minor gaps
    PARTIAL = "partial"        # Docs partially relevant
    INSUFFICIENT = "insufficient"  # Docs don't answer the question
    NONE = "none"              # No relevant docs found


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""
    id: str
    content: str
    source: str
    score: float  # Similarity score 0-1
    metadata: dict = field(default_factory=dict)
    
    # Relevance assessment (filled by reflection)
    relevance_score: Optional[float] = None
    relevance_explanation: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from retrieval with reflection."""
    documents: list[RetrievedDocument]
    quality: RetrievalQuality
    coverage_score: float  # How well docs cover the question (0-1)
    gaps: list[str] = field(default_factory=list)  # Missing information
    suggested_queries: list[str] = field(default_factory=list)
    needs_web_search: bool = False


class KnowledgeBase:
    """
    Vector-based knowledge base with document storage.
    """
    
    def __init__(self, embedding_provider: BaseEmbedding):
        self.embedding_provider = embedding_provider
        self._init_store()
    
    def _init_store(self):
        """Initialize vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            print("Warning: chromadb not installed. Using in-memory fallback.")
            self._collection = None
            self._documents = []
            self._embeddings = []
    
    def add(
        self,
        content: str,
        source: str,
        doc_id: Optional[str] = None,
        metadata: dict = None,
    ) -> str:
        """Add a document to the knowledge base."""
        import hashlib
        
        doc_id = doc_id or hashlib.md5(content.encode()).hexdigest()[:16]
        metadata = metadata or {}
        metadata["source"] = source
        
        embedding = self.embedding_provider.embed(content)
        
        if self._collection:
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
            )
        else:
            self._documents.append({
                "id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
            })
        
        return doc_id
    
    def add_batch(
        self,
        documents: list[dict],
    ) -> list[str]:
        """Add multiple documents."""
        import hashlib
        
        ids = []
        contents = []
        metadatas = []
        
        for doc in documents:
            content = doc["content"]
            doc_id = doc.get("id") or hashlib.md5(content.encode()).hexdigest()[:16]
            metadata = doc.get("metadata", {})
            metadata["source"] = doc.get("source", "unknown")
            
            ids.append(doc_id)
            contents.append(content)
            metadatas.append(metadata)
        
        embeddings = self.embedding_provider.embed_batch(contents)
        
        if self._collection:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )
        else:
            for i in range(len(ids)):
                self._documents.append({
                    "id": ids[i],
                    "content": contents[i],
                    "embedding": embeddings[i],
                    "metadata": metadatas[i],
                })
        
        return ids
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict = None,
    ) -> list[RetrievedDocument]:
        """Search the knowledge base."""
        query_embedding = self.embedding_provider.embed(query)
        
        if self._collection:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )
            
            documents = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    score = 1 - distance  # Convert distance to similarity
                    
                    documents.append(RetrievedDocument(
                        id=doc_id,
                        content=results["documents"][0][i],
                        source=results["metadatas"][0][i].get("source", "unknown"),
                        score=score,
                        metadata=results["metadatas"][0][i],
                    ))
            
            return documents
        else:
            # Fallback: simple cosine similarity
            return self._fallback_search(query_embedding, k)
    
    def _fallback_search(self, query_embedding: list[float], k: int) -> list[RetrievedDocument]:
        """Fallback search for when ChromaDB is not available."""
        import math
        
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0
        
        scores = []
        for doc in self._documents:
            score = cosine_sim(query_embedding, doc["embedding"])
            scores.append((doc, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            RetrievedDocument(
                id=doc["id"],
                content=doc["content"],
                source=doc["metadata"].get("source", "unknown"),
                score=score,
                metadata=doc["metadata"],
            )
            for doc, score in scores[:k]
        ]
    
    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        if self._collection:
            return {"count": self._collection.count()}
        return {"count": len(self._documents)}


class SelfReflectiveRetriever:
    """
    Retriever with self-reflection capabilities.
    
    Evaluates whether retrieved documents actually answer the question
    and suggests improvements or fallbacks when needed.
    """
    
    REFLECTION_SYSTEM = """You are a retrieval quality evaluator. Your job is to assess whether the retrieved documents adequately answer the user's question.

Evaluate on these criteria:
1. RELEVANCE: Are the documents topically related to the question?
2. COVERAGE: Do the documents contain the information needed to answer?
3. COMPLETENESS: Are there any gaps in the retrieved information?
4. ACCURACY: Does the information seem reliable and consistent?

Respond in this exact format:
QUALITY: [excellent|good|partial|insufficient|none]
COVERAGE_SCORE: [0.0-1.0]
GAPS: [comma-separated list of missing information, or "none"]
NEEDS_WEB_SEARCH: [true|false]
EXPLANATION: [brief explanation]"""

    REFLECTION_PROMPT = """Question: {question}

Retrieved Documents:
{documents}

Evaluate if these documents adequately answer the question."""

    RELEVANCE_SYSTEM = """You are a document relevance evaluator. Rate how relevant this document is to the question on a scale of 0-1.

Respond in this format:
SCORE: [0.0-1.0]
EXPLANATION: [brief explanation]"""

    RELEVANCE_PROMPT = """Question: {question}

Document:
{document}

Rate the relevance of this document to the question."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm: BaseLLM,
        max_iterations: int = 2,
    ):
        self.kb = knowledge_base
        self.llm = llm
        self.max_iterations = max_iterations
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        reflect: bool = True,
        assess_relevance: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve documents with self-reflection.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            reflect: Whether to evaluate retrieval quality
            assess_relevance: Whether to assess individual doc relevance
        
        Returns:
            RetrievalResult with documents and quality assessment
        """
        # Initial retrieval
        documents = self.kb.search(query, k=k)
        
        if not documents:
            return RetrievalResult(
                documents=[],
                quality=RetrievalQuality.NONE,
                coverage_score=0.0,
                gaps=["No documents found in knowledge base"],
                needs_web_search=True,
            )
        
        # Assess individual document relevance
        if assess_relevance:
            documents = self._assess_relevance(query, documents)
            
            # Filter out very low relevance docs
            documents = [d for d in documents if (d.relevance_score or 0) >= 0.3]
        
        if not documents:
            return RetrievalResult(
                documents=[],
                quality=RetrievalQuality.INSUFFICIENT,
                coverage_score=0.0,
                gaps=["Retrieved documents not relevant to the question"],
                needs_web_search=True,
            )
        
        # Reflect on overall retrieval quality
        if reflect:
            reflection = self._reflect(query, documents)
            
            # If quality is insufficient, try to improve
            if reflection.quality in [RetrievalQuality.INSUFFICIENT, RetrievalQuality.PARTIAL]:
                # Try alternative queries if suggested
                if reflection.suggested_queries:
                    for alt_query in reflection.suggested_queries[:self.max_iterations]:
                        alt_docs = self.kb.search(alt_query, k=k)
                        
                        # Add new relevant docs
                        existing_ids = {d.id for d in documents}
                        for doc in alt_docs:
                            if doc.id not in existing_ids:
                                documents.append(doc)
                    
                    # Re-reflect with augmented docs
                    reflection = self._reflect(query, documents)
            
            return reflection
        
        # No reflection - just return documents with basic assessment
        return RetrievalResult(
            documents=documents,
            quality=RetrievalQuality.GOOD if documents else RetrievalQuality.NONE,
            coverage_score=sum(d.score for d in documents) / len(documents) if documents else 0,
        )
    
    def _assess_relevance(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Assess relevance of each document."""
        for doc in documents:
            prompt = self.RELEVANCE_PROMPT.format(
                question=query,
                document=doc.content[:1000],
            )
            
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system_prompt=self.RELEVANCE_SYSTEM,
                    max_tokens=100,
                    temperature=0.1,
                )
                
                # Parse response
                for line in response.content.split('\n'):
                    if line.startswith("SCORE:"):
                        try:
                            doc.relevance_score = float(line.replace("SCORE:", "").strip())
                        except ValueError:
                            doc.relevance_score = doc.score
                    elif line.startswith("EXPLANATION:"):
                        doc.relevance_explanation = line.replace("EXPLANATION:", "").strip()
                
                if doc.relevance_score is None:
                    doc.relevance_score = doc.score
                    
            except Exception:
                doc.relevance_score = doc.score
        
        return documents
    
    def _reflect(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> RetrievalResult:
        """Reflect on retrieval quality."""
        # Format documents for reflection
        docs_text = "\n\n".join([
            f"[Document {i+1}] (relevance: {doc.relevance_score or doc.score:.2f})\n"
            f"Source: {doc.source}\n"
            f"Content: {doc.content[:500]}..."
            for i, doc in enumerate(documents)
        ])
        
        prompt = self.REFLECTION_PROMPT.format(
            question=query,
            documents=docs_text,
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.REFLECTION_SYSTEM,
                max_tokens=300,
                temperature=0.1,
            )
            
            return self._parse_reflection(response.content, documents)
            
        except Exception as e:
            # Fallback assessment
            return RetrievalResult(
                documents=documents,
                quality=RetrievalQuality.GOOD if documents else RetrievalQuality.NONE,
                coverage_score=sum(d.relevance_score or d.score for d in documents) / len(documents) if documents else 0,
            )
    
    def _parse_reflection(
        self,
        response: str,
        documents: list[RetrievedDocument],
    ) -> RetrievalResult:
        """Parse reflection response."""
        quality = RetrievalQuality.GOOD
        coverage_score = 0.7
        gaps = []
        needs_web_search = False
        
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith("QUALITY:"):
                quality_str = line.replace("QUALITY:", "").strip().lower()
                try:
                    quality = RetrievalQuality(quality_str)
                except ValueError:
                    pass
            
            elif line.startswith("COVERAGE_SCORE:"):
                try:
                    coverage_score = float(line.replace("COVERAGE_SCORE:", "").strip())
                except ValueError:
                    pass
            
            elif line.startswith("GAPS:"):
                gaps_str = line.replace("GAPS:", "").strip()
                if gaps_str.lower() != "none":
                    gaps = [g.strip() for g in gaps_str.split(",")]
            
            elif line.startswith("NEEDS_WEB_SEARCH:"):
                needs_web_search = "true" in line.lower()
        
        return RetrievalResult(
            documents=documents,
            quality=quality,
            coverage_score=coverage_score,
            gaps=gaps,
            needs_web_search=needs_web_search,
        )
