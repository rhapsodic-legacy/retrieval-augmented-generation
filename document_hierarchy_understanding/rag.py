"""
Main Hierarchical RAG orchestrator.

Ties together all components:
- Document parsing and indexing
- Hierarchical vector storage
- Cross-reference detection
- Multi-level summarization
- Context-aware retrieval
- LLM-based answer generation
"""

from pathlib import Path
from typing import Optional
import json

import anthropic

from .models import (
    HierarchyNode, Document, NodeType, QueryResult, RetrievalResult
)
from .parser import DocumentParser, MultiDocumentParser
from .vector_store import HierarchicalVectorStore
from .cross_references import CrossReferenceDetector
from .summarizer import HierarchicalSummarizer, QueryAwareSummarizer
from .retriever import HierarchicalRetriever, AdaptiveRetriever, RetrievalConfig


class HierarchicalRAG:
    """
    Production-ready Hierarchical RAG system.
    
    Features:
    - Document → Section → Paragraph hierarchy
    - Parent-child retrieval with context expansion
    - Cross-document linking and references
    - Summarization layers for efficient retrieval
    
    Usage:
        rag = HierarchicalRAG()
        rag.add_document("path/to/doc.md")
        rag.build_index()
        result = rag.query("What is the main argument?")
    """
    
    ANSWER_PROMPT = """You are a helpful assistant answering questions based on the provided context.
Use the hierarchical context to understand both the specific details and broader context.

{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Reference the document structure when relevant (e.g., "In Section X...")
- If the context doesn't contain enough information, say so
- Be concise but complete

Answer:"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        embedding_model: str = "voyage-3",
        auto_summarize: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the Hierarchical RAG system.
        
        Args:
            persist_directory: Directory to persist vector store (None for in-memory)
            model: Claude model for generation
            embedding_model: Model for embeddings (using Anthropic's Voyage)
            auto_summarize: Automatically generate summaries on index build
            verbose: Print progress information
        """
        self.model = model
        self.embedding_model = embedding_model
        self.auto_summarize = auto_summarize
        self.verbose = verbose
        
        # Initialize components
        self.client = anthropic.Anthropic()
        self.parser = MultiDocumentParser()
        self.vector_store = HierarchicalVectorStore(persist_directory)
        self.summarizer = HierarchicalSummarizer(model=model)
        self.query_summarizer = QueryAwareSummarizer(model=model)
        
        # These are initialized after documents are added
        self.cross_ref_detector: Optional[CrossReferenceDetector] = None
        self.retriever: Optional[HierarchicalRetriever] = None
        self.adaptive_retriever: Optional[AdaptiveRetriever] = None
        
        # Track state
        self._indexed = False
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(message)
    
    def _get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for text.
        
        Uses a simple character-based hash for demo purposes.
        In production, replace with actual embedding API call.
        """
        # For production, use an actual embedding API like:
        # response = self.client.embeddings.create(input=text, model=self.embedding_model)
        # return response.data[0].embedding
        
        # Simple demo embedding (replace in production!)
        import hashlib
        import struct
        
        # Create a deterministic pseudo-embedding from text hash
        text_hash = hashlib.sha256(text.encode()).digest()
        # Create 384-dimensional embedding from hash
        embedding = []
        for i in range(0, min(len(text_hash), 32), 4):
            chunk = text_hash[i:i+4]
            if len(chunk) == 4:
                val = struct.unpack('f', chunk)[0]
                # Normalize to reasonable range
                normalized = max(-1, min(1, val / 1e38)) if abs(val) > 1e-38 else 0
                embedding.append(normalized)
        
        # Pad to 384 dimensions
        while len(embedding) < 384:
            embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])
        
        return embedding[:384]
    
    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        return [self._get_embedding(text) for text in texts]
    
    def add_document(self, file_path: str) -> Document:
        """
        Add a document from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            The parsed Document object
        """
        self._log(f"📄 Adding document: {file_path}")
        doc = self.parser.add_file(file_path)
        self._log(f"   ├── Sections: {doc.total_sections}")
        self._log(f"   └── Paragraphs: {doc.total_paragraphs}")
        self._indexed = False
        return doc
    
    def add_text(self, text: str, title: str = "Untitled") -> Document:
        """
        Add a document from text content.
        
        Args:
            text: The document text
            title: Document title
            
        Returns:
            The parsed Document object
        """
        self._log(f"📄 Adding document: {title}")
        doc = self.parser.add_text(text, title)
        self._log(f"   ├── Sections: {doc.total_sections}")
        self._log(f"   └── Paragraphs: {doc.total_paragraphs}")
        self._indexed = False
        return doc
    
    def add_documents_from_directory(
        self,
        directory: str,
        extensions: list[str] = [".md", ".txt"]
    ) -> list[Document]:
        """
        Add all documents from a directory.
        
        Args:
            directory: Path to directory
            extensions: File extensions to include
            
        Returns:
            List of parsed Documents
        """
        documents = []
        dir_path = Path(directory)
        
        for ext in extensions:
            for file_path in dir_path.glob(f"**/*{ext}"):
                try:
                    doc = self.add_document(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    self._log(f"⚠️  Failed to parse {file_path}: {e}")
        
        return documents
    
    def build_index(self, generate_summaries: bool = None):
        """
        Build the vector index from all added documents.
        
        Args:
            generate_summaries: Override auto_summarize setting
        """
        if generate_summaries is None:
            generate_summaries = self.auto_summarize
        
        all_nodes = self.parser.all_nodes
        
        if not all_nodes:
            self._log("⚠️  No documents to index")
            return
        
        self._log(f"\n🔨 Building index for {len(all_nodes)} nodes...")
        
        # Step 1: Detect cross-references
        self._log("🔗 Detecting cross-references...")
        self.cross_ref_detector = CrossReferenceDetector(all_nodes)
        references = self.cross_ref_detector.process_all_references()
        total_refs = sum(len(refs) for refs in references.values())
        resolved_refs = sum(
            sum(1 for r in refs if r.resolved) 
            for refs in references.values()
        )
        self._log(f"   └── Found {total_refs} references, resolved {resolved_refs}")
        
        # Step 2: Generate summaries (optional)
        if generate_summaries:
            self._log("📝 Generating summaries...")
            for doc_id, doc in self.parser.documents.items():
                self.summarizer.summarize_hierarchy(all_nodes, doc, verbose=False)
        
        # Step 3: Generate embeddings and index
        self._log("🧮 Generating embeddings...")
        nodes_list = list(all_nodes.values())
        
        # Content embeddings
        content_texts = [n.content for n in nodes_list]
        content_embeddings = self._get_embeddings_batch(content_texts)
        
        # Summary embeddings (where available)
        summary_embeddings = []
        for node in nodes_list:
            if node.summary:
                summary_embeddings.append(self._get_embedding(node.summary))
            else:
                summary_embeddings.append(None)
        
        # Add to vector store
        self._log("💾 Indexing vectors...")
        self.vector_store.add_nodes_batch(
            nodes_list,
            content_embeddings,
            summary_embeddings
        )
        
        # Step 4: Initialize retriever
        self.retriever = HierarchicalRetriever(
            self.vector_store,
            all_nodes,
            self.cross_ref_detector
        )
        self.adaptive_retriever = AdaptiveRetriever(self.retriever)
        
        self._indexed = True
        stats = self.vector_store.get_stats()
        self._log(f"✅ Index built: {stats['total_content_nodes']} content nodes, "
                  f"{stats['total_summaries']} summaries")
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        use_adaptive: bool = True,
        include_sources: bool = True
    ) -> QueryResult:
        """
        Query the RAG system.
        
        Args:
            question: The user's question
            n_results: Number of retrieval results
            use_adaptive: Use adaptive retrieval strategy
            include_sources: Include source citations in answer
            
        Returns:
            QueryResult with answer and retrieval results
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        self._log(f"\n🔍 Query: {question}")
        
        # Get query embedding
        query_embedding = self._get_embedding(question)
        
        # Retrieve relevant content
        if use_adaptive and self.adaptive_retriever:
            retrieval_results = self.adaptive_retriever.retrieve(
                question, query_embedding, n_results
            )
        else:
            retrieval_results = self.retriever.retrieve(
                query_embedding,
                config=RetrievalConfig(n_content_results=n_results)
            )
        
        self._log(f"   └── Retrieved {len(retrieval_results)} results")
        
        # Build context for LLM
        context_parts = []
        citations = []
        
        for i, result in enumerate(retrieval_results):
            context_parts.append(f"[Source {i+1}]\n{result.full_context}")
            
            # Build citation
            doc_title = ""
            section_title = ""
            for parent in result.parent_context:
                if parent.node_type == NodeType.DOCUMENT:
                    doc_title = parent.title
                elif parent.node_type == NodeType.SECTION:
                    section_title = parent.title
            
            citation = f"[{i+1}] {doc_title}"
            if section_title:
                citation += f" > {section_title}"
            citations.append(citation)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        prompt = self.ANSWER_PROMPT.format(
            context=context,
            question=question
        )
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.content[0].text.strip()
        
        # Add citations if requested
        if include_sources and citations:
            answer += "\n\nSources:\n" + "\n".join(citations)
        
        return QueryResult(
            query=question,
            answer=answer,
            retrieval_results=retrieval_results,
            citations=citations,
            confidence=sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
        )
    
    def query_with_drill_down(
        self,
        question: str,
        broad_query: Optional[str] = None
    ) -> QueryResult:
        """
        Two-stage query: broad search then specific.
        
        Useful when you want to first identify relevant documents/sections,
        then drill down to specific content.
        
        Args:
            question: The specific question
            broad_query: Optional broader query for first stage
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        broad = broad_query or question
        
        self._log(f"\n🔍 Broad query: {broad}")
        
        # Stage 1: Find relevant documents/sections via summaries
        broad_embedding = self._get_embedding(broad)
        summary_results = self.vector_store.query_summaries(
            broad_embedding,
            n_results=3
        )
        
        if not summary_results:
            return self.query(question)
        
        # Stage 2: Query within relevant documents
        self._log(f"   └── Found {len(summary_results)} relevant sections")
        
        doc_ids = list(set(
            r["metadata"].get("document_id") 
            for r in summary_results 
            if r["metadata"].get("document_id")
        ))
        
        specific_embedding = self._get_embedding(question)
        retrieval_results = self.retriever.retrieve(
            specific_embedding,
            config=RetrievalConfig(
                target_document_ids=doc_ids,
                n_content_results=5
            )
        )
        
        # Generate answer (same as regular query)
        context_parts = []
        citations = []
        
        for i, result in enumerate(retrieval_results):
            context_parts.append(f"[Source {i+1}]\n{result.full_context}")
            citations.append(f"[{i+1}] {result.node.metadata.get('context_path', '')}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = self.ANSWER_PROMPT.format(
            context=context,
            question=question
        )
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return QueryResult(
            query=question,
            answer=response.content[0].text.strip(),
            retrieval_results=retrieval_results,
            query_type="drill_down",
            citations=citations
        )
    
    def get_document_summary(self, doc_id: str) -> Optional[str]:
        """Get the summary of a specific document."""
        doc = self.parser.get_document(doc_id)
        return doc.summary if doc else None
    
    def get_section_tree(self, doc_id: str) -> dict:
        """Get the section tree structure of a document."""
        doc = self.parser.get_document(doc_id)
        if not doc:
            return {}
        
        def build_tree(node_id: str) -> dict:
            node = self.parser.all_nodes.get(node_id)
            if not node:
                return {}
            
            tree = {
                "id": node.id,
                "title": node.title,
                "type": node.node_type.value,
                "summary": node.summary[:100] + "..." if len(node.summary) > 100 else node.summary,
                "children": []
            }
            
            for child_id in node.children_ids:
                child_tree = build_tree(child_id)
                if child_tree:
                    tree["children"].append(child_tree)
            
            return tree
        
        return build_tree(doc_id)
    
    def export_index(self, file_path: str):
        """Export the index metadata to JSON (for debugging/inspection)."""
        data = {
            "documents": {
                doc_id: {
                    "title": doc.title,
                    "sections": doc.total_sections,
                    "paragraphs": doc.total_paragraphs,
                    "summary": doc.summary
                }
                for doc_id, doc in self.parser.documents.items()
            },
            "stats": self.vector_store.get_stats(),
            "node_count": len(self.parser.all_nodes)
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._log(f"📁 Index exported to {file_path}")
