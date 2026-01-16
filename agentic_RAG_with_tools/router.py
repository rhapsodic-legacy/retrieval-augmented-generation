"""
Query Router for Agentic RAG

Classifies queries to determine the best action:
- DIRECT: Answer from LLM knowledge
- RETRIEVAL: Search knowledge base
- TOOL_USE: Use a specific tool
- WEB_SEARCH: Search the web
- HYBRID: Combine retrieval + tool use
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import re

from ..providers.llm import BaseLLM


class RouteType(Enum):
    DIRECT = "direct"           # Answer directly from LLM
    RETRIEVAL = "retrieval"     # Search knowledge base
    TOOL_USE = "tool_use"       # Use a specific tool
    WEB_SEARCH = "web_search"   # Search the web
    HYBRID = "hybrid"           # Combine approaches


@dataclass
class RouteDecision:
    """Decision from the router."""
    route_type: RouteType
    confidence: float  # 0-1
    reasoning: str
    suggested_tools: list[str] = field(default_factory=list)
    search_query: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class QueryRouter:
    """
    Routes queries to the appropriate handler.
    
    Uses pattern matching and LLM classification.
    """
    
    # Patterns for quick classification
    CALCULATION_PATTERNS = [
        r'\b\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic arithmetic
        r'\bcalculate\b',
        r'\bcompute\b',
        r'\bwhat is \d+',
        r'\bhow much is\b',
        r'\bsolve\b.*\bequation\b',
        r'\bsqrt\b|\bsquare root\b',
        r'\bpercentage\b|\b%\b',
    ]
    
    CODE_PATTERNS = [
        r'\bwrite\b.*\bcode\b',
        r'\bpython\b.*\bscript\b',
        r'\bprogram\b.*\bto\b',
        r'\bexecute\b.*\bcode\b',
        r'\brun\b.*\bpython\b',
    ]
    
    CURRENT_INFO_PATTERNS = [
        r'\bcurrent\b',
        r'\btoday\b',
        r'\brecent\b',
        r'\blatest\b',
        r'\bnow\b',
        r'\bthis week\b',
        r'\bthis month\b',
        r'\b202[3-9]\b',  # Recent years
    ]
    
    DATETIME_PATTERNS = [
        r'\bwhat time\b',
        r'\bwhat date\b',
        r'\bwhat day\b',
        r'\bcurrent time\b',
        r'\bcurrent date\b',
    ]
    
    RETRIEVAL_PATTERNS = [
        r'\baccording to\b',
        r'\bfind\b.*\bdocument\b',
        r'\bsearch\b.*\bknowledge\b',
        r'\bin my\b.*\bfiles\b',
        r'\bwhat did\b.*\bsay about\b',
        r'\blook up\b',
    ]
    
    SYSTEM_PROMPT = """You are a query classifier for an AI assistant. Your job is to analyze user queries and determine the best way to handle them.

Classification categories:
1. DIRECT - Simple questions that can be answered from general knowledge (facts, definitions, explanations)
2. RETRIEVAL - Questions about specific documents, knowledge bases, or stored information
3. TOOL_USE - Requests requiring calculations, code execution, or specific tools
4. WEB_SEARCH - Questions about current events, recent news, or real-time information
5. HYBRID - Complex questions requiring multiple approaches

Respond in this exact format:
ROUTE: [category]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
TOOLS: [comma-separated list if TOOL_USE, otherwise "none"]
SEARCH_QUERY: [optimized search query if RETRIEVAL or WEB_SEARCH, otherwise "none"]"""

    CLASSIFICATION_PROMPT = """Classify this query:

Query: {query}

Available tools: {tools}

Remember to respond in the exact format specified."""

    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize router with optional LLM for complex classification."""
        self.llm = llm
    
    def route(
        self,
        query: str,
        available_tools: list[str] = None,
        has_knowledge_base: bool = True,
    ) -> RouteDecision:
        """
        Route a query to the appropriate handler.
        
        Args:
            query: User query
            available_tools: List of available tool names
            has_knowledge_base: Whether a knowledge base is available
        
        Returns:
            RouteDecision with routing information
        """
        available_tools = available_tools or []
        query_lower = query.lower()
        
        # Try pattern-based routing first (fast path)
        pattern_decision = self._pattern_route(query_lower, available_tools)
        if pattern_decision and pattern_decision.confidence >= 0.8:
            return pattern_decision
        
        # Use LLM for complex classification
        if self.llm:
            llm_decision = self._llm_route(query, available_tools)
            
            # Merge with pattern decision if available
            if pattern_decision:
                # Use higher confidence decision
                if llm_decision.confidence > pattern_decision.confidence:
                    return llm_decision
                return pattern_decision
            
            return llm_decision
        
        # Fallback: if pattern gave something, use it
        if pattern_decision:
            return pattern_decision
        
        # Default to retrieval if knowledge base exists, else direct
        if has_knowledge_base:
            return RouteDecision(
                route_type=RouteType.RETRIEVAL,
                confidence=0.5,
                reasoning="Default to retrieval (no strong patterns matched)",
                search_query=query,
            )
        
        return RouteDecision(
            route_type=RouteType.DIRECT,
            confidence=0.5,
            reasoning="Default to direct answer (no knowledge base)",
        )
    
    def _pattern_route(
        self,
        query_lower: str,
        available_tools: list[str],
    ) -> Optional[RouteDecision]:
        """Pattern-based routing for common cases."""
        
        # Check for calculation patterns
        if "calculator" in available_tools:
            for pattern in self.CALCULATION_PATTERNS:
                if re.search(pattern, query_lower):
                    return RouteDecision(
                        route_type=RouteType.TOOL_USE,
                        confidence=0.9,
                        reasoning="Query contains mathematical expression or calculation request",
                        suggested_tools=["calculator"],
                    )
        
        # Check for datetime patterns
        if "datetime" in available_tools:
            for pattern in self.DATETIME_PATTERNS:
                if re.search(pattern, query_lower):
                    return RouteDecision(
                        route_type=RouteType.TOOL_USE,
                        confidence=0.9,
                        reasoning="Query asks for current date/time",
                        suggested_tools=["datetime"],
                    )
        
        # Check for code patterns
        if "code_executor" in available_tools:
            for pattern in self.CODE_PATTERNS:
                if re.search(pattern, query_lower):
                    return RouteDecision(
                        route_type=RouteType.TOOL_USE,
                        confidence=0.85,
                        reasoning="Query requests code execution",
                        suggested_tools=["code_executor"],
                    )
        
        # Check for current information patterns (web search)
        if "web_search" in available_tools:
            for pattern in self.CURRENT_INFO_PATTERNS:
                if re.search(pattern, query_lower):
                    return RouteDecision(
                        route_type=RouteType.WEB_SEARCH,
                        confidence=0.8,
                        reasoning="Query asks about current/recent information",
                        suggested_tools=["web_search"],
                        search_query=query_lower,
                    )
        
        # Check for retrieval patterns
        for pattern in self.RETRIEVAL_PATTERNS:
            if re.search(pattern, query_lower):
                return RouteDecision(
                    route_type=RouteType.RETRIEVAL,
                    confidence=0.85,
                    reasoning="Query explicitly references documents or knowledge base",
                    search_query=query_lower,
                )
        
        return None
    
    def _llm_route(
        self,
        query: str,
        available_tools: list[str],
    ) -> RouteDecision:
        """Use LLM for complex query classification."""
        prompt = self.CLASSIFICATION_PROMPT.format(
            query=query,
            tools=", ".join(available_tools) if available_tools else "none",
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.1,
            )
            
            return self._parse_llm_response(response.content, available_tools)
            
        except Exception as e:
            # Fallback on error
            return RouteDecision(
                route_type=RouteType.RETRIEVAL,
                confidence=0.5,
                reasoning=f"LLM classification failed: {str(e)}, defaulting to retrieval",
                search_query=query,
            )
    
    def _parse_llm_response(
        self,
        response: str,
        available_tools: list[str],
    ) -> RouteDecision:
        """Parse LLM classification response."""
        lines = response.strip().split('\n')
        
        route_type = RouteType.RETRIEVAL
        confidence = 0.7
        reasoning = "Parsed from LLM response"
        tools = []
        search_query = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("ROUTE:"):
                route_str = line.replace("ROUTE:", "").strip().upper()
                try:
                    route_type = RouteType[route_str]
                except KeyError:
                    pass
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            
            elif line.startswith("TOOLS:"):
                tools_str = line.replace("TOOLS:", "").strip()
                if tools_str.lower() != "none":
                    tools = [t.strip() for t in tools_str.split(",")]
                    # Filter to available tools
                    tools = [t for t in tools if t in available_tools]
            
            elif line.startswith("SEARCH_QUERY:"):
                sq = line.replace("SEARCH_QUERY:", "").strip()
                if sq.lower() != "none":
                    search_query = sq
        
        return RouteDecision(
            route_type=route_type,
            confidence=confidence,
            reasoning=reasoning,
            suggested_tools=tools,
            search_query=search_query,
        )


class IntentClassifier:
    """
    Classifies the intent of a query for more nuanced routing.
    """
    
    INTENTS = {
        "factual": ["what is", "who is", "where is", "when did", "how many"],
        "procedural": ["how to", "how do i", "steps to", "guide for"],
        "comparative": ["compare", "difference between", "vs", "better than"],
        "analytical": ["why", "explain", "analyze", "reason for"],
        "creative": ["write", "create", "generate", "compose"],
        "computational": ["calculate", "compute", "solve", "evaluate"],
        "retrieval": ["find", "search", "look up", "get me"],
    }
    
    @classmethod
    def classify(cls, query: str) -> tuple[str, float]:
        """
        Classify query intent.
        
        Returns:
            (intent_name, confidence)
        """
        query_lower = query.lower()
        
        best_intent = "unknown"
        best_score = 0
        
        for intent, patterns in cls.INTENTS.items():
            score = sum(1 for p in patterns if p in query_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        confidence = min(best_score / 2, 1.0)  # Normalize
        return best_intent, confidence
