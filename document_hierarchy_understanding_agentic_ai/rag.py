"""
Hierarchical RAG with AutoGen Multi-Agent System

This is the main entry point for the AutoGen-based hierarchical RAG.
It provides a simple interface while using multi-agent collaboration under the hood.
"""

from typing import Optional
from dataclasses import dataclass

from .core.parser import MultiDocumentParser
from .core.vector_store import HierarchicalVectorStore
from .core.models import Document, QueryResult
from .tools import (
    initialize_tools,
    add_document,
    add_text_document,
    list_documents,
    get_document_structure,
    search_content,
    search_summaries,
    get_node_content,
    get_parent_context,
    get_siblings,
    find_references_in_node,
    get_related_nodes,
    build_index,
    get_index_stats,
)
from .orchestrator import (
    HierarchicalRAGGroupChat,
    SequentialRAGWorkflow,
    TwoAgentRAG,
)


@dataclass
class RAGConfig:
    """Configuration for the AutoGen RAG system."""
    model: str = "claude-sonnet-4-20250514"
    persist_directory: Optional[str] = None
    max_rounds: int = 15
    verbose: bool = True
    workflow_type: str = "group_chat"  # "group_chat", "sequential", "simple"


class HierarchicalRAGAutoGen:
    """
    Hierarchical Document RAG powered by AutoGen multi-agent system.
    
    This system uses multiple specialized agents:
    - Coordinator: Understands queries and orchestrates workflow
    - Retriever: Searches vector store with hierarchy awareness
    - CrossReference: Detects and follows document references
    - Summarizer: Synthesizes information from multiple sources
    - Verifier: Checks accuracy and citations
    
    Usage:
        rag = HierarchicalRAGAutoGen()
        rag.add_document("path/to/doc.md")
        rag.build_index()
        result = rag.query("What is the main argument?")
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the AutoGen RAG system.
        
        Args:
            config: Configuration options
        """
        self.config = config or RAGConfig()
        
        # Initialize core components
        self.parser, self.vector_store = initialize_tools(
            self.config.persist_directory
        )
        
        # Build tool function dictionary
        self.tool_functions = {
            "add_document": add_document,
            "add_text_document": add_text_document,
            "list_documents": list_documents,
            "get_document_structure": get_document_structure,
            "search_content": search_content,
            "search_summaries": search_summaries,
            "get_node_content": get_node_content,
            "get_parent_context": get_parent_context,
            "get_siblings": get_siblings,
            "find_references_in_node": find_references_in_node,
            "get_related_nodes": get_related_nodes,
            "build_index": build_index,
            "get_index_stats": get_index_stats,
        }
        
        # Initialize workflow based on config
        self.workflow = None
        self._indexed = False
    
    def _ensure_workflow(self):
        """Lazily initialize the workflow when needed."""
        if self.workflow is None:
            if self.config.workflow_type == "group_chat":
                self.workflow = HierarchicalRAGGroupChat(
                    self.tool_functions,
                    model=self.config.model,
                    max_rounds=self.config.max_rounds,
                    verbose=self.config.verbose,
                )
            elif self.config.workflow_type == "sequential":
                self.workflow = SequentialRAGWorkflow(
                    self.tool_functions,
                    model=self.config.model,
                    verbose=self.config.verbose,
                )
            else:  # simple
                self.workflow = TwoAgentRAG(
                    self.tool_functions,
                    model=self.config.model,
                )
    
    def add_document(self, file_path: str) -> Document:
        """
        Add a document to the knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            The parsed Document object
        """
        if self.config.verbose:
            print(f"📄 Adding document: {file_path}")
        
        doc = self.parser.add_file(file_path)
        
        if self.config.verbose:
            print(f"   ├── Sections: {doc.total_sections}")
            print(f"   └── Paragraphs: {doc.total_paragraphs}")
        
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
        if self.config.verbose:
            print(f"📄 Adding document: {title}")
        
        doc = self.parser.add_text(text, title)
        
        if self.config.verbose:
            print(f"   ├── Sections: {doc.total_sections}")
            print(f"   └── Paragraphs: {doc.total_paragraphs}")
        
        self._indexed = False
        return doc
    
    def add_documents_from_directory(
        self,
        directory: str,
        extensions: list[str] = [".md", ".txt"]
    ) -> list[Document]:
        """Add all documents from a directory."""
        from pathlib import Path
        
        documents = []
        dir_path = Path(directory)
        
        for ext in extensions:
            for file_path in dir_path.glob(f"**/*{ext}"):
                try:
                    doc = self.add_document(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    if self.config.verbose:
                        print(f"⚠️  Failed to parse {file_path}: {e}")
        
        return documents
    
    def build_index(self):
        """
        Build the vector index from all added documents.
        """
        if self.config.verbose:
            print("\n🔨 Building index...")
        
        nodes = list(self.parser.all_nodes.values())
        
        if not nodes:
            print("⚠️  No documents to index")
            return
        
        # Generate simple embeddings (replace in production)
        embeddings = [self._get_simple_embedding(n.content) for n in nodes]
        summary_embeddings = [
            self._get_simple_embedding(n.summary) if n.summary else None
            for n in nodes
        ]
        
        # Index
        self.vector_store.add_nodes_batch(nodes, embeddings, summary_embeddings)
        
        self._indexed = True
        
        if self.config.verbose:
            stats = self.vector_store.get_stats()
            print(f"✅ Indexed {stats['total_content_nodes']} nodes")
    
    def query(self, question: str) -> str:
        """
        Query the RAG system using multi-agent collaboration.
        
        Args:
            question: The user's question
            
        Returns:
            The answer from the agent collaboration
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        self._ensure_workflow()
        
        if self.config.verbose:
            print(f"\n🔍 Query: {question}")
            print(f"🤖 Using {self.config.workflow_type} workflow...")
            print("-" * 50)
        
        if isinstance(self.workflow, SequentialRAGWorkflow):
            result = self.workflow.query(question)
            return result["answer"]
        else:
            return self.workflow.query(question)
    
    def query_with_details(self, question: str) -> dict:
        """
        Query with full details about the agent collaboration.
        
        Returns:
            Dictionary with answer and collaboration details
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        self._ensure_workflow()
        
        if isinstance(self.workflow, SequentialRAGWorkflow):
            return self.workflow.query(question)
        elif isinstance(self.workflow, HierarchicalRAGGroupChat):
            answer = self.workflow.query(question)
            return {
                "answer": answer,
                "conversation": self.workflow.get_conversation_history()
            }
        else:
            return {"answer": self.workflow.query(question)}
    
    def get_document_tree(self, doc_id: str) -> dict:
        """Get the section tree of a document."""
        doc = self.parser.get_document(doc_id)
        if not doc:
            return {}
        
        def build_tree(node_id: str) -> dict:
            node = self.parser.all_nodes.get(node_id)
            if not node:
                return {}
            
            return {
                "id": node.id,
                "title": node.title,
                "type": node.node_type.value,
                "children": [
                    build_tree(child_id)
                    for child_id in node.children_ids
                ]
            }
        
        return build_tree(doc_id)
    
    def _get_simple_embedding(self, text: str) -> list[float]:
        """Simple embedding for demo. Replace in production."""
        import hashlib
        import struct
        
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = []
        
        for i in range(0, min(len(text_hash), 32), 4):
            chunk = text_hash[i:i+4]
            if len(chunk) == 4:
                val = struct.unpack('f', chunk)[0]
                normalized = max(-1, min(1, val / 1e38)) if abs(val) > 1e-38 else 0
                embedding.append(normalized)
        
        while len(embedding) < 384:
            embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])
        
        return embedding[:384]


# Convenience function
def create_rag(
    workflow_type: str = "simple",
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = True
) -> HierarchicalRAGAutoGen:
    """
    Create a hierarchical RAG instance.
    
    Args:
        workflow_type: "group_chat", "sequential", or "simple"
        model: LLM model to use
        verbose: Print progress
        
    Returns:
        Configured RAG instance
    """
    config = RAGConfig(
        model=model,
        workflow_type=workflow_type,
        verbose=verbose,
    )
    return HierarchicalRAGAutoGen(config)
