"""
Code Repository RAG

Intelligent Q&A over codebases with:
- AST-aware chunking
- Dependency graph
- Multi-file tracing
- Test association
"""

from dataclasses import dataclass, field
from typing import Optional, Generator
from pathlib import Path
import os
import re

from .providers import create_llm, create_embedding, BaseLLM, BaseEmbedding
from .parsing import (
    parse_file,
    is_supported_file,
    is_test_file,
    CodeUnit,
    CodeUnitType,
    ParsedFile,
)
from .graph import DependencyGraph, GraphBuilder, RelationType
from .indexing import CodeIndex, IndexedUnit, SearchResult


@dataclass
class CodeRAGConfig:
    """Configuration for Code RAG."""
    
    # LLM settings
    llm_provider: str = "anthropic"
    llm_model: Optional[str] = None
    
    # Embedding settings
    embedding_provider: str = "local"  # "voyage" recommended for production
    embedding_model: Optional[str] = None
    
    # Search settings
    n_results: int = 10
    include_tests: bool = True
    include_related: bool = True  # Include related code via graph
    max_related_depth: int = 2
    
    # Generation settings
    max_tokens: int = 2000
    temperature: float = 0.1
    
    # File filtering
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "node_modules", "__pycache__", ".git", "dist", "build",
        "*.min.js", "*.bundle.js", "*.pyc",
    ])


@dataclass
class CodeContext:
    """Context retrieved for a query."""
    primary_results: list[SearchResult]
    related_code: list[IndexedUnit]
    test_code: list[IndexedUnit]
    call_chain: Optional[dict] = None
    files_involved: list[str] = field(default_factory=list)


@dataclass
class CodeRAGResponse:
    """Response from Code RAG."""
    answer: str
    context: CodeContext
    query: str
    model: str
    
    # Trace information
    code_flow: Optional[list[dict]] = None
    metadata: dict = field(default_factory=dict)


class CodeRAG:
    """
    Code Repository RAG System.
    
    Features:
    - AST-aware code parsing
    - Dependency graph for code relationships
    - Multi-file code tracing
    - Semantic + symbol search
    - Test association
    
    Usage:
        rag = CodeRAG()
        rag.index_directory("./my_project")
        
        response = rag.query("How does the authentication flow work?")
        print(response.answer)
        
        # Trace code flow
        flow = rag.trace_flow("login", "create_session")
    """
    
    SYSTEM_PROMPT = """You are an expert code analyst helping developers understand codebases.

When answering questions:
1. Reference specific functions, classes, and files
2. Explain the code flow step by step
3. Highlight important patterns and relationships
4. Mention relevant tests if available
5. Use code snippets to illustrate points

Format code blocks with syntax highlighting:
```python
def example():
    pass
```

Be thorough but concise. Focus on the "why" not just the "what"."""

    QUERY_PROMPT = """Based on the following code context, answer the question.

{context}

QUESTION: {question}

Provide a detailed explanation referencing the relevant code. Include code snippets where helpful."""

    TRACE_PROMPT = """Trace the code flow from {start} to {end} based on this code:

{context}

Explain step-by-step how execution flows from the starting point to the endpoint.
Include function calls, data transformations, and any important side effects."""

    def __init__(self, config: Optional[CodeRAGConfig] = None):
        """Initialize Code RAG."""
        self.config = config or CodeRAGConfig()
        
        # Initialize providers
        self.llm = create_llm(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )
        
        self.embeddings = create_embedding(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
        )
        
        # Initialize components
        self.index = CodeIndex(embedding_provider=self.embeddings)
        self.graph = DependencyGraph()
        self.graph_builder: Optional[GraphBuilder] = None
        
        # File tracking
        self.indexed_files: set[str] = set()
        self.base_path: Optional[str] = None
    
    # =========================================================================
    # Indexing
    # =========================================================================
    
    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> dict:
        """
        Index a code directory.
        
        Args:
            directory: Path to directory
            recursive: Include subdirectories
            
        Returns:
            Indexing statistics
        """
        directory = Path(directory).resolve()
        self.base_path = str(directory)
        
        # Find all supported files
        files = []
        
        if recursive:
            for path in directory.rglob("*"):
                if self._should_index(path):
                    files.append(str(path))
        else:
            for path in directory.iterdir():
                if path.is_file() and self._should_index(path):
                    files.append(str(path))
        
        # Parse and index
        stats = {"files": 0, "units": 0, "errors": 0}
        parsed_files = []
        
        for file_path in files:
            try:
                parsed = parse_file(file_path)
                parsed_files.append(parsed)
                
                # Index code units
                self.index.index_file(parsed)
                self.indexed_files.add(file_path)
                
                stats["files"] += 1
                stats["units"] += len(parsed.units)
                
            except Exception as e:
                print(f"Warning: Failed to index {file_path}: {e}")
                stats["errors"] += 1
        
        # Build dependency graph
        self.graph_builder = GraphBuilder(self.base_path)
        self.graph = self.graph_builder.build(list(self.indexed_files))
        
        return stats
    
    def index_file(self, file_path: str) -> dict:
        """Index a single file."""
        file_path = str(Path(file_path).resolve())
        
        if not self._should_index(Path(file_path)):
            return {"status": "skipped", "reason": "excluded by pattern"}
        
        try:
            parsed = parse_file(file_path)
            self.index.index_file(parsed)
            self.indexed_files.add(file_path)
            
            # Update graph
            if self.graph_builder:
                # Rebuild graph with new file
                self.graph = self.graph_builder.build(list(self.indexed_files))
            
            return {
                "status": "indexed",
                "units": len(parsed.units),
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _should_index(self, path: Path) -> bool:
        """Check if a file should be indexed."""
        if not path.is_file():
            return False
        
        if not is_supported_file(str(path)):
            return False
        
        # Check exclude patterns
        path_str = str(path)
        for pattern in self.config.exclude_patterns:
            if '*' in pattern:
                # Glob pattern
                import fnmatch
                if fnmatch.fnmatch(path.name, pattern):
                    return False
            else:
                # Directory pattern
                if pattern in path_str:
                    return False
        
        return True
    
    # =========================================================================
    # Query
    # =========================================================================
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> CodeRAGResponse:
        """
        Query the codebase.
        
        Args:
            question: Natural language question
            k: Number of results
            
        Returns:
            CodeRAGResponse with answer and context
        """
        k = k or self.config.n_results
        
        # Retrieve context
        context = self._retrieve_context(question, k)
        
        # Build context string
        context_str = self._format_context(context)
        
        # Generate answer
        prompt = self.QUERY_PROMPT.format(
            context=context_str,
            question=question,
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        return CodeRAGResponse(
            answer=response.content,
            context=context,
            query=question,
            model=response.model,
        )
    
    def query_stream(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> Generator[str, None, CodeRAGResponse]:
        """Stream a query response."""
        k = k or self.config.n_results
        
        context = self._retrieve_context(question, k)
        context_str = self._format_context(context)
        
        prompt = self.QUERY_PROMPT.format(
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
        
        return CodeRAGResponse(
            answer=full_answer,
            context=context,
            query=question,
            model=getattr(self.llm, 'model', 'unknown'),
        )
    
    def _retrieve_context(self, question: str, k: int) -> CodeContext:
        """Retrieve relevant code context for a question."""
        # Primary search
        primary_results = self.index.search_hybrid(question, k=k)
        
        # Get related code via dependency graph
        related_code = []
        if self.config.include_related:
            for result in primary_results[:5]:
                related = self.graph.find_related_code(
                    result.unit.id,
                    max_depth=self.config.max_related_depth,
                )
                for node, distance in related:
                    unit = self.index.get_unit(node.id)
                    if unit and unit not in [r.unit for r in primary_results]:
                        related_code.append(unit)
        
        # Get associated tests
        test_code = []
        if self.config.include_tests:
            for result in primary_results[:5]:
                test_nodes = self.graph.get_tests_for(result.unit.id)
                for node in test_nodes:
                    unit = self.index.get_unit(node.id)
                    if unit:
                        test_code.append(unit)
        
        # Collect files involved
        files = set()
        for result in primary_results:
            files.add(result.unit.file_path)
        for unit in related_code + test_code:
            files.add(unit.file_path)
        
        return CodeContext(
            primary_results=primary_results,
            related_code=related_code[:10],
            test_code=test_code[:5],
            files_involved=list(files),
        )
    
    def _format_context(self, context: CodeContext) -> str:
        """Format context for the prompt."""
        parts = []
        
        # Primary results
        parts.append("=== RELEVANT CODE ===\n")
        for i, result in enumerate(context.primary_results, 1):
            parts.append(self._format_unit(result.unit, i))
        
        # Related code
        if context.related_code:
            parts.append("\n=== RELATED CODE ===\n")
            for unit in context.related_code[:5]:
                parts.append(self._format_unit(unit))
        
        # Test code
        if context.test_code:
            parts.append("\n=== RELATED TESTS ===\n")
            for unit in context.test_code:
                parts.append(self._format_unit(unit))
        
        return "\n".join(parts)
    
    def _format_unit(self, unit: IndexedUnit, num: Optional[int] = None) -> str:
        """Format a code unit for the prompt."""
        parts = []
        
        # Header
        header = f"[{num}] " if num else ""
        header += f"{unit.type.value.upper()}: {unit.name}"
        if unit.parent_class:
            header = f"{header} (in class {unit.parent_class})"
        parts.append(header)
        
        # File location
        parts.append(f"File: {unit.file_path}:{unit.start_line}")
        
        # Docstring
        if unit.docstring:
            parts.append(f"Docstring: {unit.docstring[:200]}...")
        
        # Source code
        parts.append("```")
        # Limit source length
        source = unit.source
        if len(source) > 1000:
            source = source[:1000] + "\n... (truncated)"
        parts.append(source)
        parts.append("```\n")
        
        return "\n".join(parts)
    
    # =========================================================================
    # Code Flow Tracing
    # =========================================================================
    
    def trace_flow(
        self,
        start: str,
        end: str,
    ) -> CodeRAGResponse:
        """
        Trace the code flow between two points.
        
        Args:
            start: Starting function/method name
            end: Ending function/method name
            
        Returns:
            CodeRAGResponse with traced flow
        """
        # Find start and end units
        start_results = self.index.search_symbol(start, exact=False)
        end_results = self.index.search_symbol(end, exact=False)
        
        if not start_results or not end_results:
            return CodeRAGResponse(
                answer=f"Could not find '{start}' or '{end}' in the codebase.",
                context=CodeContext(
                    primary_results=[],
                    related_code=[],
                    test_code=[],
                ),
                query=f"Trace flow: {start} → {end}",
                model="",
            )
        
        start_unit = start_results[0].unit
        end_unit = end_results[0].unit
        
        # Find path in call graph
        path = self.graph.find_call_path(start_unit.id, end_unit.id)
        
        # Collect code along the path
        path_units = [start_unit]
        if path:
            for node, rel in path:
                unit = self.index.get_unit(node.id)
                if unit:
                    path_units.append(unit)
        
        # Build context
        context = CodeContext(
            primary_results=[SearchResult(u, 1.0, "trace") for u in path_units],
            related_code=[],
            test_code=[],
            call_chain=self.graph.get_call_chain(start_unit.id, direction="down"),
        )
        
        context_str = self._format_context(context)
        
        # Generate trace explanation
        prompt = self.TRACE_PROMPT.format(
            start=start,
            end=end,
            context=context_str,
        )
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        # Build flow diagram
        code_flow = []
        for i, unit in enumerate(path_units):
            code_flow.append({
                "step": i + 1,
                "name": unit.name,
                "type": unit.type.value,
                "file": unit.file_path,
                "signature": unit.signature,
            })
        
        return CodeRAGResponse(
            answer=response.content,
            context=context,
            query=f"Trace flow: {start} → {end}",
            model=response.model,
            code_flow=code_flow,
        )
    
    # =========================================================================
    # Symbol Lookup
    # =========================================================================
    
    def find_symbol(self, name: str) -> list[dict]:
        """Find a symbol by name."""
        results = self.index.search_symbol(name, exact=False)
        
        return [
            {
                "id": r.unit.id,
                "name": r.unit.name,
                "type": r.unit.type.value,
                "file": r.unit.file_path,
                "line": r.unit.start_line,
                "signature": r.unit.signature,
                "docstring": r.unit.docstring,
            }
            for r in results
        ]
    
    def get_callers(self, name: str) -> list[dict]:
        """Get all functions that call a given function."""
        results = self.index.search_symbol(name, exact=True)
        
        callers = []
        for result in results:
            caller_nodes = self.graph.get_callers(result.unit.id)
            for node in caller_nodes:
                unit = self.index.get_unit(node.id)
                if unit:
                    callers.append({
                        "name": unit.name,
                        "type": unit.type.value,
                        "file": unit.file_path,
                        "line": unit.start_line,
                    })
        
        return callers
    
    def get_callees(self, name: str) -> list[dict]:
        """Get all functions called by a given function."""
        results = self.index.search_symbol(name, exact=True)
        
        callees = []
        for result in results:
            callee_nodes = self.graph.get_callees(result.unit.id)
            for node in callee_nodes:
                unit = self.index.get_unit(node.id)
                if unit:
                    callees.append({
                        "name": unit.name,
                        "type": unit.type.value,
                        "file": unit.file_path,
                        "line": unit.start_line,
                    })
        
        return callees
    
    def get_tests_for(self, name: str) -> list[dict]:
        """Get tests for a given function/class."""
        results = self.index.search_symbol(name, exact=True)
        
        tests = []
        for result in results:
            test_nodes = self.graph.get_tests_for(result.unit.id)
            for node in test_nodes:
                unit = self.index.get_unit(node.id)
                if unit:
                    tests.append({
                        "name": unit.name,
                        "file": unit.file_path,
                        "line": unit.start_line,
                        "source": unit.source[:500],
                    })
        
        return tests
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "indexed_files": len(self.indexed_files),
            "index": self.index.get_stats(),
            "graph": self.graph.get_stats(),
            "config": {
                "llm_provider": self.config.llm_provider,
                "embedding_provider": self.config.embedding_provider,
            },
        }
    
    def get_file_summary(self, file_path: str) -> dict:
        """Get a summary of a file."""
        units = self.index.get_units_in_file(file_path)
        
        classes = [u for u in units if u.type == CodeUnitType.CLASS]
        functions = [u for u in units if u.type in (
            CodeUnitType.FUNCTION,
            CodeUnitType.ASYNC_FUNCTION,
        )]
        methods = [u for u in units if u.type == CodeUnitType.METHOD]
        
        return {
            "file": file_path,
            "classes": [
                {"name": c.name, "docstring": c.docstring}
                for c in classes
            ],
            "functions": [
                {"name": f.name, "signature": f.signature, "docstring": f.docstring}
                for f in functions
            ],
            "methods": [
                {"name": m.name, "class": m.parent_class, "signature": m.signature}
                for m in methods
            ],
            "imports": self.graph.get_imports(file_path),
            "importers": self.graph.get_importers(file_path),
        }
