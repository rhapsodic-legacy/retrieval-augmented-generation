"""
Hierarchy-aware retrieval with context expansion.

Implements the key RAG strategies:
1. Summary-first retrieval (query summaries to find relevant docs/sections)
2. Paragraph retrieval with parent context expansion
3. Cross-reference expansion
4. Multi-hop retrieval for complex queries
"""

from typing import Optional
from dataclasses import dataclass

from .models import (
    HierarchyNode, NodeType, RetrievalResult, Document, Section, Paragraph
)
from .vector_store import HierarchicalVectorStore
from .cross_references import CrossReferenceDetector


@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior."""
    # Number of results at each stage
    n_summary_results: int = 3
    n_content_results: int = 5
    
    # Context expansion
    include_parent_context: bool = True
    include_sibling_context: bool = False
    max_sibling_count: int = 2
    
    # Cross-reference expansion
    include_references: bool = True
    max_reference_depth: int = 1
    max_references: int = 3
    
    # Summary layer
    use_summary_layer: bool = True
    summary_threshold: float = 0.5  # Min score to consider summary match
    
    # Filtering
    target_node_types: Optional[list[NodeType]] = None
    target_document_ids: Optional[list[str]] = None


class HierarchicalRetriever:
    """
    Retriever that understands document hierarchy.
    
    Key features:
    - Two-stage retrieval: summaries first, then drill down
    - Parent context expansion: returns paragraph + section + document context
    - Cross-reference expansion: includes related nodes
    """
    
    def __init__(
        self,
        vector_store: HierarchicalVectorStore,
        all_nodes: dict[str, HierarchyNode],
        cross_ref_detector: Optional[CrossReferenceDetector] = None,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: The hierarchical vector store
            all_nodes: Dictionary of all nodes
            cross_ref_detector: Optional cross-reference detector
            config: Retrieval configuration
        """
        self.vector_store = vector_store
        self.all_nodes = all_nodes
        self.cross_ref_detector = cross_ref_detector
        self.config = config or RetrievalConfig()
    
    def retrieve(
        self,
        query_embedding: list[float],
        config: Optional[RetrievalConfig] = None
    ) -> list[RetrievalResult]:
        """
        Perform hierarchical retrieval.
        
        Strategy:
        1. If use_summary_layer: Query summaries to identify relevant docs/sections
        2. Query content (optionally filtered by summary matches)
        3. Expand with parent context
        4. Expand with cross-references
        
        Args:
            query_embedding: The query vector
            config: Optional override config
            
        Returns:
            List of retrieval results with expanded context
        """
        cfg = config or self.config
        results = []
        
        # Stage 1: Summary layer (optional)
        relevant_doc_ids = None
        if cfg.use_summary_layer:
            summary_matches = self.vector_store.query_summaries(
                query_embedding,
                n_results=cfg.n_summary_results,
                node_types=[NodeType.DOCUMENT, NodeType.SECTION]
            )
            
            # Filter to high-scoring summary matches
            relevant_doc_ids = [
                m["metadata"].get("document_id")
                for m in summary_matches
                if m["score"] >= cfg.summary_threshold
            ]
            relevant_doc_ids = list(set(filter(None, relevant_doc_ids)))
        
        # Stage 2: Content retrieval
        content_results = self.vector_store.query(
            query_embedding,
            n_results=cfg.n_content_results,
            node_types=cfg.target_node_types or [NodeType.PARAGRAPH, NodeType.SECTION],
            document_ids=relevant_doc_ids if relevant_doc_ids else cfg.target_document_ids
        )
        
        # Stage 3: Build retrieval results with context expansion
        for result in content_results:
            node_id = result["id"]
            node = self.all_nodes.get(node_id)
            
            if not node:
                continue
            
            retrieval_result = RetrievalResult(
                node=node,
                score=result["score"]
            )
            
            # Expand parent context
            if cfg.include_parent_context:
                retrieval_result.parent_context = self._get_parent_chain(node_id)
            
            # Expand sibling context
            if cfg.include_sibling_context:
                retrieval_result.sibling_context = self._get_siblings(
                    node_id, cfg.max_sibling_count
                )
            
            # Expand cross-references
            if cfg.include_references and self.cross_ref_detector:
                retrieval_result.related_nodes = self._get_related_nodes(
                    node_id, cfg.max_reference_depth, cfg.max_references
                )
            
            # Build full context string
            retrieval_result.build_full_context()
            
            results.append(retrieval_result)
        
        return results
    
    def retrieve_with_summary_drill_down(
        self,
        query_embedding: list[float],
        n_summaries: int = 3,
        n_per_summary: int = 2
    ) -> list[RetrievalResult]:
        """
        Two-stage retrieval: find relevant summaries, then drill down.
        
        This is useful for broad queries where you want to find the most
        relevant documents/sections first, then retrieve specific content.
        
        Args:
            query_embedding: Query vector
            n_summaries: Number of summary matches
            n_per_summary: Number of content pieces per summary match
            
        Returns:
            Retrieval results organized by matched summary
        """
        results = []
        
        # Stage 1: Query summaries
        summary_matches = self.vector_store.query_summaries(
            query_embedding,
            n_results=n_summaries
        )
        
        # Stage 2: For each summary match, retrieve relevant content
        for summary_match in summary_matches:
            node_id = summary_match["metadata"].get("node_id")
            doc_id = summary_match["metadata"].get("document_id")
            
            # Get content from this document/section
            if node_id:
                # Get children of this node
                children = self.vector_store.get_children(node_id)
                
                # Score children against query (simplified - use first n)
                for child in children[:n_per_summary]:
                    child_node = self.all_nodes.get(child["id"])
                    if child_node:
                        result = RetrievalResult(
                            node=child_node,
                            score=summary_match["score"] * 0.9  # Slightly lower than parent
                        )
                        result.parent_context = self._get_parent_chain(child["id"])
                        result.build_full_context()
                        results.append(result)
            elif doc_id:
                # Query within this document
                doc_results = self.vector_store.query(
                    query_embedding,
                    n_results=n_per_summary,
                    document_ids=[doc_id],
                    node_types=[NodeType.PARAGRAPH]
                )
                
                for doc_result in doc_results:
                    node = self.all_nodes.get(doc_result["id"])
                    if node:
                        result = RetrievalResult(
                            node=node,
                            score=doc_result["score"]
                        )
                        result.parent_context = self._get_parent_chain(doc_result["id"])
                        result.build_full_context()
                        results.append(result)
        
        return results
    
    def retrieve_with_cross_references(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        reference_depth: int = 1
    ) -> list[RetrievalResult]:
        """
        Retrieve with aggressive cross-reference expansion.
        
        Useful for queries that might span multiple related documents.
        """
        # Get initial results
        initial_results = self.retrieve(
            query_embedding,
            config=RetrievalConfig(
                n_content_results=n_results,
                include_references=False  # We'll handle this ourselves
            )
        )
        
        if not self.cross_ref_detector:
            return initial_results
        
        # Collect all referenced nodes
        all_node_ids = set()
        for result in initial_results:
            all_node_ids.add(result.node.id)
            
            # Get referenced nodes
            related = self.cross_ref_detector.get_related_nodes(
                result.node.id, max_depth=reference_depth
            )
            all_node_ids.update(related)
        
        # Build results for all nodes
        results = []
        seen_ids = set()
        
        # Add initial results first (they have scores)
        for result in initial_results:
            results.append(result)
            seen_ids.add(result.node.id)
        
        # Add referenced nodes that weren't in initial results
        for node_id in all_node_ids:
            if node_id in seen_ids:
                continue
            
            node = self.all_nodes.get(node_id)
            if node:
                result = RetrievalResult(
                    node=node,
                    score=0.5  # Lower score for reference-based results
                )
                result.parent_context = self._get_parent_chain(node_id)
                result.build_full_context()
                results.append(result)
                seen_ids.add(node_id)
        
        return results
    
    def _get_parent_chain(self, node_id: str) -> list[HierarchyNode]:
        """Get the full parent chain from root to parent of node."""
        chain = []
        current = self.all_nodes.get(node_id)
        
        if not current:
            return chain
        
        # Walk up to root
        parent_id = current.parent_id
        while parent_id:
            parent = self.all_nodes.get(parent_id)
            if parent:
                chain.append(parent)
                parent_id = parent.parent_id
            else:
                break
        
        return chain[::-1]  # Reverse to get root-to-parent order
    
    def _get_siblings(self, node_id: str, max_count: int) -> list[HierarchyNode]:
        """Get sibling nodes (same parent)."""
        node = self.all_nodes.get(node_id)
        if not node or not node.parent_id:
            return []
        
        parent = self.all_nodes.get(node.parent_id)
        if not parent:
            return []
        
        siblings = []
        node_position = node.position
        
        for child_id in parent.children_ids:
            if child_id == node_id:
                continue
            
            child = self.all_nodes.get(child_id)
            if child:
                # Prefer adjacent siblings
                distance = abs(child.position - node_position)
                siblings.append((distance, child))
        
        # Sort by distance and return closest
        siblings.sort(key=lambda x: x[0])
        return [s[1] for s in siblings[:max_count]]
    
    def _get_related_nodes(
        self,
        node_id: str,
        max_depth: int,
        max_count: int
    ) -> list[HierarchyNode]:
        """Get nodes related through cross-references."""
        if not self.cross_ref_detector:
            return []
        
        related_ids = self.cross_ref_detector.get_related_nodes(node_id, max_depth)
        related_nodes = []
        
        for related_id in related_ids[:max_count]:
            node = self.all_nodes.get(related_id)
            if node:
                related_nodes.append(node)
        
        return related_nodes


class AdaptiveRetriever:
    """
    Retriever that adapts strategy based on query characteristics.
    
    - Broad queries → summary-first approach
    - Specific queries → direct content search
    - Cross-reference queries → reference expansion
    """
    
    def __init__(self, base_retriever: HierarchicalRetriever):
        self.retriever = base_retriever
    
    def classify_query(self, query: str) -> str:
        """
        Classify query type based on characteristics.
        
        Returns: "broad", "specific", or "cross-reference"
        """
        query_lower = query.lower()
        
        # Cross-reference indicators
        cross_ref_terms = ["related to", "connection", "linked", "referenced", "see also"]
        if any(term in query_lower for term in cross_ref_terms):
            return "cross-reference"
        
        # Broad query indicators
        broad_terms = ["overview", "summary", "what is", "explain", "describe", "main"]
        if any(term in query_lower for term in broad_terms):
            return "broad"
        
        # Specific query indicators (questions with specific details)
        specific_terms = ["how", "when", "where", "which", "specific", "exact", "detail"]
        if any(term in query_lower for term in specific_terms):
            return "specific"
        
        # Default to specific for most queries
        return "specific"
    
    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        n_results: int = 5
    ) -> list[RetrievalResult]:
        """
        Retrieve using adaptive strategy based on query type.
        """
        query_type = self.classify_query(query)
        
        if query_type == "broad":
            return self.retriever.retrieve_with_summary_drill_down(
                query_embedding,
                n_summaries=3,
                n_per_summary=2
            )
        elif query_type == "cross-reference":
            return self.retriever.retrieve_with_cross_references(
                query_embedding,
                n_results=n_results,
                reference_depth=2
            )
        else:  # specific
            return self.retriever.retrieve(
                query_embedding,
                config=RetrievalConfig(
                    n_content_results=n_results,
                    use_summary_layer=True,
                    include_parent_context=True,
                    include_references=True
                )
            )
