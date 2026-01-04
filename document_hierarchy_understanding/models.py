"""
Data models for hierarchical document structure.

Defines the Document → Section → Paragraph hierarchy with support for
cross-document references and summaries at each level.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import uuid
import hashlib


class NodeType(Enum):
    """Types of nodes in the document hierarchy."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"


@dataclass
class Reference:
    """A reference/citation to another document or section."""
    source_node_id: str
    target_node_id: Optional[str]  # None if unresolved
    target_document_id: Optional[str]
    reference_text: str  # The original reference text (e.g., "See Section 3.2")
    reference_type: str  # "citation", "cross-reference", "footnote", etc.
    resolved: bool = False
    
    def to_dict(self) -> dict:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "target_document_id": self.target_document_id,
            "reference_text": self.reference_text,
            "reference_type": self.reference_type,
            "resolved": self.resolved
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Reference":
        return cls(**data)


@dataclass
class HierarchyNode:
    """
    Base node in the document hierarchy.
    
    Each node has:
    - Unique ID and content
    - Parent/child relationships
    - Optional summary
    - References to other nodes/documents
    """
    id: str
    node_type: NodeType
    content: str
    title: str = ""
    summary: str = ""
    
    # Hierarchy relationships
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    
    # Document membership
    document_id: str = ""
    
    # Position in document
    level: int = 0  # Depth in hierarchy (0 = document, 1 = section, 2+ = subsection/paragraph)
    position: int = 0  # Order among siblings
    
    # Cross-references
    references: list[Reference] = field(default_factory=list)
    referenced_by: list[str] = field(default_factory=list)  # Node IDs that reference this node
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    
    @staticmethod
    def generate_id(content: str, node_type: NodeType) -> str:
        """Generate a unique ID based on content hash and UUID."""
        content_hash = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        unique_id = str(uuid.uuid4())[:8]
        return f"{node_type.value}_{content_hash}_{unique_id}"
    
    def get_context_path(self) -> str:
        """Returns the hierarchical path for context (e.g., 'Doc > Section > Subsection')."""
        return self.metadata.get("context_path", self.title)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "content": self.content,
            "title": self.title,
            "summary": self.summary,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "document_id": self.document_id,
            "level": self.level,
            "position": self.position,
            "references": [r.to_dict() for r in self.references],
            "referenced_by": self.referenced_by,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "HierarchyNode":
        data["node_type"] = NodeType(data["node_type"])
        data["references"] = [Reference.from_dict(r) for r in data.get("references", [])]
        return cls(**data)


@dataclass
class Document(HierarchyNode):
    """
    Top-level document node.
    
    Contains metadata about the entire document and serves as the root
    of the hierarchy tree.
    """
    source_path: str = ""
    total_sections: int = 0
    total_paragraphs: int = 0
    
    def __post_init__(self):
        self.node_type = NodeType.DOCUMENT
        self.level = 0
        if not self.id:
            self.id = self.generate_id(self.content, NodeType.DOCUMENT)
        self.document_id = self.id


@dataclass 
class Section(HierarchyNode):
    """
    Section node (can be nested for subsections).
    
    Sections can contain other sections (subsections) and/or paragraphs.
    """
    section_number: str = ""  # e.g., "3.2.1"
    
    def __post_init__(self):
        self.node_type = NodeType.SECTION
        if not self.id:
            self.id = self.generate_id(self.content, NodeType.SECTION)


@dataclass
class Paragraph(HierarchyNode):
    """
    Paragraph node (leaf nodes in the hierarchy).
    
    Paragraphs are the atomic units for retrieval but include
    parent context when returned.
    """
    
    def __post_init__(self):
        self.node_type = NodeType.PARAGRAPH
        if not self.id:
            self.id = self.generate_id(self.content, NodeType.PARAGRAPH)


@dataclass
class RetrievalResult:
    """
    Result from hierarchical retrieval.
    
    Includes the matched node plus expanded context from parent nodes.
    """
    node: HierarchyNode
    score: float
    
    # Expanded context
    parent_context: list[HierarchyNode] = field(default_factory=list)
    sibling_context: list[HierarchyNode] = field(default_factory=list)
    
    # Cross-references
    related_nodes: list[HierarchyNode] = field(default_factory=list)
    
    # Full context string for LLM
    full_context: str = ""
    
    def build_full_context(self) -> str:
        """Build the full context string including hierarchy."""
        parts = []
        
        # Add parent context (document -> section -> subsection)
        for parent in self.parent_context:
            if parent.node_type == NodeType.DOCUMENT:
                parts.append(f"📄 Document: {parent.title}")
                if parent.summary:
                    parts.append(f"   Summary: {parent.summary}")
            elif parent.node_type == NodeType.SECTION:
                indent = "  " * parent.level
                parts.append(f"{indent}📑 Section: {parent.title}")
                if parent.summary:
                    parts.append(f"{indent}   Summary: {parent.summary}")
        
        # Add the main content
        parts.append(f"\n{'='*50}")
        parts.append(f"📝 Matched Content (relevance: {self.score:.2f}):")
        parts.append(self.node.content)
        parts.append(f"{'='*50}\n")
        
        # Add related cross-references
        if self.related_nodes:
            parts.append("🔗 Related References:")
            for related in self.related_nodes[:3]:  # Limit to top 3
                parts.append(f"  - [{related.node_type.value}] {related.title}: {related.content[:200]}...")
        
        self.full_context = "\n".join(parts)
        return self.full_context


@dataclass
class QueryResult:
    """Complete result from a RAG query."""
    query: str
    answer: str
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    
    # Query routing info
    query_type: str = ""  # "summary", "detail", "cross-reference"
    reformulated_query: str = ""
    
    # Summary layer results
    summary_matches: list[tuple[str, float]] = field(default_factory=list)  # (doc_id, score)
    
    # Confidence and citations
    confidence: float = 0.0
    citations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "query_type": self.query_type,
            "confidence": self.confidence,
            "citations": self.citations,
            "num_sources": len(self.retrieval_results)
        }
