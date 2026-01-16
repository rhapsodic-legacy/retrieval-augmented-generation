"""
Agentic RAG - Smart Assistant with Tool Use

Main orchestrator that:
- Routes queries to appropriate handlers
- Uses tools when needed
- Retrieves from knowledge base with self-reflection
- Falls back to web search when needed
- Generates answers with citations and confidence scores
"""

from dataclasses import dataclass, field
from typing import Optional, Generator, Callable
import time

from .providers import create_llm, create_embedding, BaseLLM, BaseEmbedding
from .routing.router import QueryRouter, RouteDecision, RouteType
from .retrieval.retriever import (
    KnowledgeBase,
    SelfReflectiveRetriever,
    RetrievalResult,
    RetrievalQuality,
)
from .tools.tool_system import ToolRegistry, ToolResult, BaseTool
from .generation.generator import ResponseGenerator, GeneratedResponse, Citation, SourceType


@dataclass
class AgenticRAGConfig:
    """Configuration for Agentic RAG."""
    
    # LLM settings
    llm_provider: str = "gemini"  # "gemini", "anthropic", "openai"
    llm_model: Optional[str] = None
    
    # Embedding settings
    embedding_provider: str = "local"
    embedding_model: Optional[str] = None
    
    # Retrieval settings
    retrieval_k: int = 5
    enable_reflection: bool = True
    enable_relevance_assessment: bool = True
    
    # Routing settings
    enable_routing: bool = True
    
    # Tool settings
    enable_tools: bool = True
    enable_web_search: bool = True
    enable_code_execution: bool = True
    
    # Generation settings
    max_tokens: int = 2000
    temperature: float = 0.1
    assess_confidence: bool = True


@dataclass
class AgentStep:
    """A step in the agent's reasoning process."""
    step_type: str  # "routing", "retrieval", "tool", "generation"
    description: str
    result: Optional[str] = None
    duration_ms: int = 0


@dataclass
class AgenticResponse:
    """Complete response from the Agentic RAG system."""
    answer: str
    confidence: float
    citations: list[Citation]
    source_types: list[SourceType]
    
    # Execution trace
    route_decision: Optional[RouteDecision] = None
    steps: list[AgentStep] = field(default_factory=list)
    
    # Tool results
    tool_results: list[tuple[str, ToolResult]] = field(default_factory=list)
    
    # Retrieval results
    retrieval_result: Optional[RetrievalResult] = None
    
    # Web search results
    web_results: Optional[list[dict]] = None
    
    # Metadata
    total_time_ms: int = 0
    model_used: str = ""


class AgenticRAG:
    """
    Agentic RAG with Tool Use.
    
    A smart assistant that decides when to:
    - Answer directly from LLM knowledge
    - Retrieve from knowledge base
    - Use tools (calculator, code, etc.)
    - Search the web
    
    Features:
    - Intelligent query routing
    - Self-reflective retrieval
    - Multiple tool support
    - Citations with confidence scores
    
    Usage:
        rag = AgenticRAG()
        
        # Add documents to knowledge base
        rag.add_document("content...", source="doc.pdf")
        
        # Query with automatic routing
        response = rag.query("What is 25 * 4?")
        # → Uses calculator tool
        
        response = rag.query("What does our policy say about refunds?")
        # → Retrieves from knowledge base
        
        response = rag.query("What happened in the news today?")
        # → Falls back to web search
    """
    
    def __init__(self, config: Optional[AgenticRAGConfig] = None):
        """Initialize Agentic RAG."""
        self.config = config or AgenticRAGConfig()
        
        # Initialize LLM
        self.llm = create_llm(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )
        
        # Initialize embeddings
        self.embeddings = create_embedding(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
        )
        
        # Initialize components
        self.knowledge_base = KnowledgeBase(self.embeddings)
        self.retriever = SelfReflectiveRetriever(
            self.knowledge_base,
            self.llm,
        )
        self.router = QueryRouter(self.llm if self.config.enable_routing else None)
        self.generator = ResponseGenerator(self.llm)
        
        # Initialize tools
        self.tool_registry = ToolRegistry()
        
        # Disable tools based on config
        if not self.config.enable_web_search:
            self.tool_registry.unregister("web_search")
        if not self.config.enable_code_execution:
            self.tool_registry.unregister("code_executor")
        
        # Callbacks
        self.on_step: Optional[Callable[[AgentStep], None]] = None
    
    # =========================================================================
    # Knowledge Base Management
    # =========================================================================
    
    def add_document(
        self,
        content: str,
        source: str,
        doc_id: Optional[str] = None,
        metadata: dict = None,
    ) -> str:
        """Add a document to the knowledge base."""
        return self.knowledge_base.add(content, source, doc_id, metadata)
    
    def add_documents(self, documents: list[dict]) -> list[str]:
        """Add multiple documents."""
        return self.knowledge_base.add_batch(documents)
    
    def add_file(self, file_path: str) -> list[str]:
        """Add a file to the knowledge base."""
        import os
        
        source = os.path.basename(file_path)
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Simple chunking
        chunks = self._chunk_text(content)
        
        documents = [
            {"content": chunk, "source": source, "metadata": {"file": file_path, "chunk": i}}
            for i, chunk in enumerate(chunks)
        ]
        
        return self.add_documents(documents)
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """Simple text chunking."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                for sep in ['. ', '.\n', '\n\n', '\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return [c for c in chunks if c]
    
    # =========================================================================
    # Tool Management
    # =========================================================================
    
    def register_tool(self, tool: BaseTool):
        """Register a custom tool."""
        self.tool_registry.register(tool)
    
    def list_tools(self) -> list[str]:
        """List available tools."""
        return self.tool_registry.list_tools()
    
    # =========================================================================
    # Query Processing
    # =========================================================================
    
    def query(self, question: str) -> AgenticResponse:
        """
        Process a query with intelligent routing.
        
        Automatically decides whether to:
        - Answer directly
        - Retrieve from knowledge base
        - Use tools
        - Search the web
        """
        start_time = time.time()
        steps = []
        tool_results = []
        retrieval_result = None
        web_results = None
        
        # Step 1: Route the query
        route_step_start = time.time()
        available_tools = self.tool_registry.list_tools() if self.config.enable_tools else []
        has_kb = self.knowledge_base.get_stats()["count"] > 0
        
        route_decision = self.router.route(
            question,
            available_tools=available_tools,
            has_knowledge_base=has_kb,
        )
        
        route_step = AgentStep(
            step_type="routing",
            description=f"Routed to {route_decision.route_type.value}",
            result=route_decision.reasoning,
            duration_ms=int((time.time() - route_step_start) * 1000),
        )
        steps.append(route_step)
        self._emit_step(route_step)
        
        # Step 2: Execute based on route
        if route_decision.route_type == RouteType.TOOL_USE:
            tool_results = self._execute_tools(
                question,
                route_decision.suggested_tools,
                steps,
            )
        
        elif route_decision.route_type == RouteType.RETRIEVAL:
            retrieval_result = self._retrieve(question, steps)
            
            # Check if retrieval is sufficient
            if retrieval_result.needs_web_search and self.config.enable_web_search:
                web_results = self._web_search(question, steps)
        
        elif route_decision.route_type == RouteType.WEB_SEARCH:
            web_results = self._web_search(question, steps)
        
        elif route_decision.route_type == RouteType.HYBRID:
            # Do both retrieval and tools
            retrieval_result = self._retrieve(question, steps)
            
            if route_decision.suggested_tools:
                tool_results = self._execute_tools(
                    question,
                    route_decision.suggested_tools,
                    steps,
                )
            
            if retrieval_result.needs_web_search and self.config.enable_web_search:
                web_results = self._web_search(question, steps)
        
        # DIRECT route: no retrieval or tools needed
        
        # Step 3: Generate response
        gen_step_start = time.time()
        
        generated = self.generator.generate(
            question=question,
            retrieval_result=retrieval_result,
            tool_results=tool_results if tool_results else None,
            web_results=web_results,
            assess_confidence=self.config.assess_confidence,
        )
        
        gen_step = AgentStep(
            step_type="generation",
            description="Generated response with citations",
            result=f"Confidence: {generated.confidence:.0%}",
            duration_ms=int((time.time() - gen_step_start) * 1000),
        )
        steps.append(gen_step)
        self._emit_step(gen_step)
        
        total_time = int((time.time() - start_time) * 1000)
        
        return AgenticResponse(
            answer=generated.answer,
            confidence=generated.confidence,
            citations=generated.citations,
            source_types=generated.source_types_used,
            route_decision=route_decision,
            steps=steps,
            tool_results=tool_results,
            retrieval_result=retrieval_result,
            web_results=web_results,
            total_time_ms=total_time,
            model_used=self.llm.model_name,
        )
    
    def query_stream(self, question: str) -> Generator[str, None, AgenticResponse]:
        """Stream a query response."""
        # First do routing and retrieval
        available_tools = self.tool_registry.list_tools() if self.config.enable_tools else []
        has_kb = self.knowledge_base.get_stats()["count"] > 0
        
        route_decision = self.router.route(question, available_tools, has_kb)
        
        retrieval_result = None
        tool_results = []
        web_results = None
        
        # Execute based on route (non-streaming parts)
        if route_decision.route_type in [RouteType.RETRIEVAL, RouteType.HYBRID]:
            retrieval_result = self.retriever.retrieve(
                question,
                k=self.config.retrieval_k,
                reflect=self.config.enable_reflection,
            )
        
        if route_decision.route_type == RouteType.TOOL_USE or route_decision.suggested_tools:
            for tool_name in route_decision.suggested_tools:
                result = self._execute_single_tool(question, tool_name)
                if result:
                    tool_results.append(result)
        
        if route_decision.route_type == RouteType.WEB_SEARCH or (
            retrieval_result and retrieval_result.needs_web_search
        ):
            web_results = self._do_web_search(question)
        
        # Stream the response
        full_answer = ""
        for chunk in self.generator.generate_stream(
            question=question,
            retrieval_result=retrieval_result,
            tool_results=tool_results if tool_results else None,
            web_results=web_results,
        ):
            full_answer += chunk
            yield chunk
        
        # Return final response
        return AgenticResponse(
            answer=full_answer,
            confidence=0.7,  # Simplified for streaming
            citations=[],
            source_types=[SourceType.LLM_KNOWLEDGE],
            route_decision=route_decision,
            retrieval_result=retrieval_result,
            tool_results=tool_results,
            web_results=web_results,
            model_used=self.llm.model_name,
        )
    
    def _execute_tools(
        self,
        question: str,
        tool_names: list[str],
        steps: list[AgentStep],
    ) -> list[tuple[str, ToolResult]]:
        """Execute suggested tools."""
        results = []
        
        for tool_name in tool_names:
            result = self._execute_single_tool(question, tool_name)
            if result:
                results.append(result)
                
                step = AgentStep(
                    step_type="tool",
                    description=f"Executed {tool_name}",
                    result=result[1].to_string()[:200],
                )
                steps.append(step)
                self._emit_step(step)
        
        return results
    
    def _execute_single_tool(
        self,
        question: str,
        tool_name: str,
    ) -> Optional[tuple[str, ToolResult]]:
        """Execute a single tool, inferring arguments from the question."""
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return None
        
        # Use LLM to extract arguments
        args = self._extract_tool_args(question, tool)
        
        result = tool.execute(**args)
        return (tool_name, result)
    
    def _extract_tool_args(self, question: str, tool: BaseTool) -> dict:
        """Extract tool arguments from the question using LLM."""
        definition = tool.definition
        
        # Simple extraction for common tools
        if definition.name == "calculator":
            # Extract mathematical expression
            import re
            numbers = re.findall(r'[\d\.\+\-\*\/\(\)\^]+', question)
            if numbers:
                return {"expression": ' '.join(numbers)}
            return {"expression": question}
        
        elif definition.name == "datetime":
            if "time" in question.lower():
                return {"operation": "time"}
            elif "date" in question.lower():
                return {"operation": "date"}
            elif "day" in question.lower():
                return {"operation": "weekday"}
            return {"operation": "now"}
        
        elif definition.name == "web_search":
            return {"query": question}
        
        elif definition.name == "code_executor":
            # Try to extract code from the question
            return {"code": question}
        
        # Generic: use LLM to extract
        prompt = f"""Extract the arguments for the tool "{definition.name}" from this question:

Tool description: {definition.description}
Parameters: {definition.parameters}

Question: {question}

Return the arguments as a JSON object."""

        try:
            response = self.llm.generate(prompt, max_tokens=200, temperature=0)
            import json
            # Try to parse JSON from response
            content = response.content
            # Find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass
        
        return {}
    
    def _retrieve(
        self,
        question: str,
        steps: list[AgentStep],
    ) -> RetrievalResult:
        """Retrieve from knowledge base with reflection."""
        step_start = time.time()
        
        result = self.retriever.retrieve(
            question,
            k=self.config.retrieval_k,
            reflect=self.config.enable_reflection,
            assess_relevance=self.config.enable_relevance_assessment,
        )
        
        step = AgentStep(
            step_type="retrieval",
            description=f"Retrieved {len(result.documents)} documents (quality: {result.quality.value})",
            result=f"Coverage: {result.coverage_score:.0%}, Gaps: {result.gaps or 'none'}",
            duration_ms=int((time.time() - step_start) * 1000),
        )
        steps.append(step)
        self._emit_step(step)
        
        return result
    
    def _web_search(
        self,
        question: str,
        steps: list[AgentStep],
    ) -> Optional[list[dict]]:
        """Execute web search."""
        step_start = time.time()
        
        results = self._do_web_search(question)
        
        step = AgentStep(
            step_type="tool",
            description=f"Web search: found {len(results) if results else 0} results",
            result=results[0]["title"] if results else "No results",
            duration_ms=int((time.time() - step_start) * 1000),
        )
        steps.append(step)
        self._emit_step(step)
        
        return results
    
    def _do_web_search(self, query: str) -> Optional[list[dict]]:
        """Actually perform web search."""
        web_tool = self.tool_registry.get("web_search")
        if not web_tool:
            return None
        
        result = web_tool.execute(query=query)
        if result.success and isinstance(result.output, list):
            return result.output
        return None
    
    def _emit_step(self, step: AgentStep):
        """Emit a step to the callback if set."""
        if self.on_step:
            self.on_step(step)
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "knowledge_base": self.knowledge_base.get_stats(),
            "tools": self.tool_registry.list_tools(),
            "config": {
                "llm_provider": self.config.llm_provider,
                "embedding_provider": self.config.embedding_provider,
            },
        }
