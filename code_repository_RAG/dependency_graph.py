"""
Dependency Graph

Tracks relationships between code units:
- Import relationships (file → file)
- Call relationships (function → function)
- Inheritance (class → class)
- References (code unit → code unit)
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import defaultdict
import json


class RelationType(Enum):
    """Types of relationships between code units."""
    IMPORTS = "imports"           # File imports another file
    CALLS = "calls"               # Function/method calls another
    EXTENDS = "extends"           # Class extends another
    IMPLEMENTS = "implements"     # Class implements interface
    USES = "uses"                 # Uses a class/type
    TESTS = "tests"               # Test file tests a module
    CONTAINS = "contains"         # File contains code unit


@dataclass
class Relationship:
    """A relationship between two code entities."""
    source_id: str
    target_id: str
    type: RelationType
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphNode:
    """A node in the dependency graph."""
    id: str
    name: str
    type: str  # "file", "class", "function", "method"
    file_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class DependencyGraph:
    """
    Graph tracking dependencies and relationships in a codebase.
    
    Enables:
    - "What imports this file?"
    - "What does this function call?"
    - "What tests cover this code?"
    - "Trace the call flow from A to B"
    """
    
    def __init__(self):
        # Node storage
        self.nodes: dict[str, GraphNode] = {}
        
        # Adjacency lists
        self.outgoing: dict[str, list[Relationship]] = defaultdict(list)
        self.incoming: dict[str, list[Relationship]] = defaultdict(list)
        
        # Indexes for fast lookup
        self.nodes_by_type: dict[str, set[str]] = defaultdict(set)
        self.nodes_by_file: dict[str, set[str]] = defaultdict(set)
        self.relations_by_type: dict[RelationType, list[Relationship]] = defaultdict(list)
        
        # Symbol table for name lookups
        self.symbol_table: dict[str, list[str]] = defaultdict(list)  # name → [node_ids]
    
    # =========================================================================
    # Node operations
    # =========================================================================
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.nodes_by_type[node.type].add(node.id)
        
        if node.file_path:
            self.nodes_by_file[node.file_path].add(node.id)
        
        # Add to symbol table
        self.symbol_table[node.name].append(node.id)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def find_by_name(self, name: str) -> list[GraphNode]:
        """Find nodes by name."""
        node_ids = self.symbol_table.get(name, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_nodes_in_file(self, file_path: str) -> list[GraphNode]:
        """Get all nodes in a file."""
        node_ids = self.nodes_by_file.get(file_path, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        """Get all nodes of a type."""
        node_ids = self.nodes_by_type.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    # =========================================================================
    # Relationship operations
    # =========================================================================
    
    def add_relationship(self, rel: Relationship):
        """Add a relationship to the graph."""
        self.outgoing[rel.source_id].append(rel)
        self.incoming[rel.target_id].append(rel)
        self.relations_by_type[rel.type].append(rel)
    
    def get_outgoing(
        self,
        node_id: str,
        rel_types: Optional[list[RelationType]] = None,
    ) -> list[tuple[GraphNode, Relationship]]:
        """Get outgoing relationships from a node."""
        results = []
        
        for rel in self.outgoing.get(node_id, []):
            if rel_types and rel.type not in rel_types:
                continue
            
            target = self.nodes.get(rel.target_id)
            if target:
                results.append((target, rel))
        
        return results
    
    def get_incoming(
        self,
        node_id: str,
        rel_types: Optional[list[RelationType]] = None,
    ) -> list[tuple[GraphNode, Relationship]]:
        """Get incoming relationships to a node."""
        results = []
        
        for rel in self.incoming.get(node_id, []):
            if rel_types and rel.type not in rel_types:
                continue
            
            source = self.nodes.get(rel.source_id)
            if source:
                results.append((source, rel))
        
        return results
    
    # =========================================================================
    # Query operations
    # =========================================================================
    
    def get_callers(self, node_id: str) -> list[GraphNode]:
        """Get all nodes that call this node."""
        return [
            node for node, rel in self.get_incoming(node_id, [RelationType.CALLS])
        ]
    
    def get_callees(self, node_id: str) -> list[GraphNode]:
        """Get all nodes that this node calls."""
        return [
            node for node, rel in self.get_outgoing(node_id, [RelationType.CALLS])
        ]
    
    def get_importers(self, file_path: str) -> list[str]:
        """Get all files that import a given file."""
        file_node_id = f"file::{file_path}"
        return [
            node.file_path for node, rel in self.get_incoming(file_node_id, [RelationType.IMPORTS])
            if node.file_path
        ]
    
    def get_imports(self, file_path: str) -> list[str]:
        """Get all files imported by a given file."""
        file_node_id = f"file::{file_path}"
        return [
            node.file_path for node, rel in self.get_outgoing(file_node_id, [RelationType.IMPORTS])
            if node.file_path
        ]
    
    def get_tests_for(self, node_id: str) -> list[GraphNode]:
        """Get test nodes that test a given node."""
        return [
            node for node, rel in self.get_incoming(node_id, [RelationType.TESTS])
        ]
    
    def get_tested_by(self, test_node_id: str) -> list[GraphNode]:
        """Get nodes that a test node tests."""
        return [
            node for node, rel in self.get_outgoing(test_node_id, [RelationType.TESTS])
        ]
    
    def get_subclasses(self, class_id: str) -> list[GraphNode]:
        """Get classes that extend a given class."""
        return [
            node for node, rel in self.get_incoming(class_id, [RelationType.EXTENDS])
        ]
    
    def get_parent_class(self, class_id: str) -> Optional[GraphNode]:
        """Get the parent class of a given class."""
        parents = [
            node for node, rel in self.get_outgoing(class_id, [RelationType.EXTENDS])
        ]
        return parents[0] if parents else None
    
    # =========================================================================
    # Path finding
    # =========================================================================
    
    def find_call_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 10,
    ) -> Optional[list[tuple[GraphNode, Relationship]]]:
        """
        Find a call path from one node to another.
        
        Returns the path as a list of (node, relationship) tuples.
        """
        if from_id == to_id:
            return []
        
        # BFS
        visited = {from_id}
        queue = [(from_id, [])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            for target, rel in self.get_outgoing(current_id, [RelationType.CALLS]):
                if target.id == to_id:
                    return path + [(target, rel)]
                
                if target.id not in visited:
                    visited.add(target.id)
                    queue.append((target.id, path + [(target, rel)]))
        
        return None
    
    def get_call_chain(
        self,
        node_id: str,
        direction: str = "down",  # "up" for callers, "down" for callees
        max_depth: int = 5,
    ) -> dict:
        """
        Get the call chain from a node.
        
        Returns a tree structure of calls.
        """
        def build_tree(nid: str, depth: int, visited: set) -> dict:
            if depth >= max_depth or nid in visited:
                return None
            
            visited = visited | {nid}
            node = self.nodes.get(nid)
            
            if not node:
                return None
            
            if direction == "down":
                children = self.get_callees(nid)
            else:
                children = self.get_callers(nid)
            
            child_trees = []
            for child in children:
                child_tree = build_tree(child.id, depth + 1, visited)
                if child_tree:
                    child_trees.append(child_tree)
            
            return {
                "id": nid,
                "name": node.name,
                "type": node.type,
                "children": child_trees,
            }
        
        return build_tree(node_id, 0, set())
    
    def find_related_code(
        self,
        node_id: str,
        max_depth: int = 2,
    ) -> list[tuple[GraphNode, int]]:
        """
        Find all code related to a node within max_depth hops.
        
        Returns list of (node, distance) tuples.
        """
        results = []
        visited = {node_id}
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth > 0:
                node = self.nodes.get(current_id)
                if node:
                    results.append((node, depth))
            
            if depth >= max_depth:
                continue
            
            # Get all connected nodes
            for target, rel in self.get_outgoing(current_id):
                if target.id not in visited:
                    visited.add(target.id)
                    queue.append((target.id, depth + 1))
            
            for source, rel in self.get_incoming(current_id):
                if source.id not in visited:
                    visited.add(source.id)
                    queue.append((source.id, depth + 1))
        
        return results
    
    # =========================================================================
    # Statistics and export
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_relationships": sum(len(rels) for rels in self.outgoing.values()),
            "nodes_by_type": {
                t: len(ids) for t, ids in self.nodes_by_type.items()
            },
            "relations_by_type": {
                t.value: len(rels) for t, rels in self.relations_by_type.items()
            },
            "files": len(self.nodes_by_file),
        }
    
    def to_dict(self) -> dict:
        """Export graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.type,
                    "file_path": n.file_path,
                    "metadata": n.metadata,
                }
                for n in self.nodes.values()
            ],
            "relationships": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.type.value,
                    "weight": r.weight,
                    "metadata": r.metadata,
                }
                for rels in self.outgoing.values()
                for r in rels
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        """Import graph from dictionary."""
        graph = cls()
        
        for n in data.get("nodes", []):
            node = GraphNode(
                id=n["id"],
                name=n["name"],
                type=n["type"],
                file_path=n.get("file_path"),
                metadata=n.get("metadata", {}),
            )
            graph.add_node(node)
        
        for r in data.get("relationships", []):
            rel = Relationship(
                source_id=r["source_id"],
                target_id=r["target_id"],
                type=RelationType(r["type"]),
                weight=r.get("weight", 1.0),
                metadata=r.get("metadata", {}),
            )
            graph.add_relationship(rel)
        
        return graph
    
    def to_networkx(self):
        """Export to NetworkX graph for visualization."""
        import networkx as nx
        
        G = nx.DiGraph()
        
        for node in self.nodes.values():
            G.add_node(
                node.id,
                name=node.name,
                type=node.type,
                file_path=node.file_path,
                **node.metadata,
            )
        
        for rels in self.outgoing.values():
            for rel in rels:
                G.add_edge(
                    rel.source_id,
                    rel.target_id,
                    type=rel.type.value,
                    weight=rel.weight,
                    **rel.metadata,
                )
        
        return G
