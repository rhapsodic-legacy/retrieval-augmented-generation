"""
Conversation-Aware Query Rewriter

Handles:
- Pronoun resolution ("it", "they", "that one")
- Reference resolution ("the second one", "the previous answer")
- Context injection (adding relevant context from conversation)
- Follow-up question expansion
"""

from typing import Optional
from dataclasses import dataclass

from .providers.llm import BaseLLMProvider, Message
from .memory.manager import MemoryManager


@dataclass
class RewriteResult:
    """Result of query rewriting."""
    original_query: str
    rewritten_query: str
    was_rewritten: bool
    confidence: float
    resolved_references: list[dict]  # What was resolved
    reasoning: str


class QueryRewriter:
    """
    Rewrites queries to be self-contained using conversation context.
    
    Examples:
        "What about the second one?" → "What about the second document mentioned, 'Sales Report Q3'?"
        "Tell me more" → "Tell me more about the authentication flow in Section 3.2"
        "How does it work?" → "How does the caching mechanism described in the API documentation work?"
    """
    
    REWRITE_PROMPT = """You are a query rewriter for a document Q&A system. Your job is to rewrite user queries to be self-contained and unambiguous.

CONVERSATION CONTEXT:
{context}

RECENT ENTITIES MENTIONED:
{entities}

ACTIVE CITATIONS/SOURCES:
{citations}

USER'S NEW QUERY:
{query}

TASK:
If the query contains:
- Pronouns (it, they, this, that, these, those)
- References (the first one, the second, the previous, that document)
- Implicit context (tell me more, continue, what else, how about)
- Comparisons without clear subjects (what's the difference, which is better)

Then rewrite the query to be fully self-contained, replacing all references with their actual referents.

If the query is already clear and self-contained, return it unchanged.

RESPOND IN THIS JSON FORMAT:
{{
  "rewritten_query": "The fully self-contained query",
  "was_rewritten": true/false,
  "confidence": 0.0-1.0,
  "resolved_references": [
    {{"original": "it", "resolved_to": "the authentication system", "source": "previous message"}}
  ],
  "reasoning": "Brief explanation of changes made"
}}

Only rewrite when necessary. Maintain the user's intent and tone."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        memory_manager: MemoryManager,
    ):
        """
        Initialize the query rewriter.
        
        Args:
            llm_provider: LLM to use for rewriting
            memory_manager: Memory manager for context
        """
        self.llm = llm_provider
        self.memory = memory_manager
    
    def needs_rewriting(self, query: str) -> bool:
        """
        Quick check if a query likely needs rewriting.
        
        Uses heuristics before invoking LLM.
        """
        query_lower = query.lower()
        
        # Pronouns that need resolution
        pronouns = ["it", "they", "this", "that", "these", "those", "its", "their"]
        
        # Reference phrases
        references = [
            "the first", "the second", "the third", "the last",
            "the previous", "the next", "the other",
            "that one", "this one", "the same",
            "mentioned earlier", "discussed before",
            "the document", "the section", "the paragraph",
        ]
        
        # Continuation phrases
        continuations = [
            "tell me more", "more about", "continue",
            "what else", "anything else", "go on",
            "elaborate", "expand on", "more details",
            "what about", "how about",
        ]
        
        # Comparison without subject
        comparisons = [
            "what's the difference", "which is better",
            "compare them", "versus", " vs ",
        ]
        
        # Check pronouns at word boundaries
        words = query_lower.split()
        for pronoun in pronouns:
            if pronoun in words:
                return True
        
        # Check phrases
        for phrase in references + continuations + comparisons:
            if phrase in query_lower:
                return True
        
        # Very short queries often need context
        if len(query.split()) <= 3 and "?" in query:
            return True
        
        return False
    
    def rewrite(self, query: str, force: bool = False) -> RewriteResult:
        """
        Rewrite a query to be self-contained.
        
        Args:
            query: The user's query
            force: Force rewriting even if heuristics say no
            
        Returns:
            RewriteResult with original and rewritten query
        """
        # Quick check
        if not force and not self.needs_rewriting(query):
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                was_rewritten=False,
                confidence=1.0,
                resolved_references=[],
                reasoning="Query is already self-contained",
            )
        
        # Build context for rewriting
        context = self._build_context()
        entities = self._build_entities_context()
        citations = self._build_citations_context()
        
        # Call LLM
        prompt = self.REWRITE_PROMPT.format(
            context=context,
            entities=entities,
            citations=citations,
            query=query,
        )
        
        response = self.llm.generate(
            messages=[Message(role="user", content=prompt)],
            max_tokens=500,
            temperature=0.1,
        )
        
        # Parse response
        return self._parse_rewrite_response(query, response.content)
    
    def _build_context(self) -> str:
        """Build conversation context string."""
        messages = self.memory.get_conversation_context(5)
        
        if not messages:
            return "No previous conversation."
        
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"][:300]
            if len(msg["content"]) > 300:
                content += "..."
            parts.append(f"{role}: {content}")
        
        return "\n".join(parts)
    
    def _build_entities_context(self) -> str:
        """Build entities context string."""
        entities = self.memory.get_recent_entities(10)
        
        if not entities:
            return "No entities tracked."
        
        parts = []
        for entity in entities:
            parts.append(f"- {entity.name} ({entity.type}): first mentioned as \"{entity.first_mention}\"")
        
        return "\n".join(parts)
    
    def _build_citations_context(self) -> str:
        """Build citations context string."""
        citations = self.memory.get_recent_citations(5)
        
        if not citations:
            return "No sources cited yet."
        
        parts = []
        for i, citation in enumerate(citations, 1):
            parts.append(f"[{i}] {citation.source}: {citation.content[:150]}...")
        
        return "\n".join(parts)
    
    def _parse_rewrite_response(self, original: str, response: str) -> RewriteResult:
        """Parse the LLM response into a RewriteResult."""
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                return RewriteResult(
                    original_query=original,
                    rewritten_query=data.get("rewritten_query", original),
                    was_rewritten=data.get("was_rewritten", False),
                    confidence=data.get("confidence", 0.5),
                    resolved_references=data.get("resolved_references", []),
                    reasoning=data.get("reasoning", ""),
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback: return original
        return RewriteResult(
            original_query=original,
            rewritten_query=original,
            was_rewritten=False,
            confidence=0.5,
            resolved_references=[],
            reasoning="Failed to parse LLM response",
        )
    
    def rewrite_with_fallback(self, query: str) -> str:
        """
        Simple rewrite that returns just the rewritten query string.
        
        Falls back to original if rewriting fails.
        """
        result = self.rewrite(query)
        return result.rewritten_query


class RuleBasedRewriter:
    """
    Simple rule-based rewriter for faster processing.
    
    Use this for simple cases, fall back to LLM for complex ones.
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
    
    def rewrite(self, query: str) -> Optional[str]:
        """
        Try to rewrite using rules. Returns None if LLM needed.
        """
        query_lower = query.lower()
        
        # Handle "tell me more" type queries
        if any(phrase in query_lower for phrase in ["tell me more", "more about this", "elaborate"]):
            # Get last topic
            entities = self.memory.get_recent_entities(1)
            if entities:
                return f"Tell me more about {entities[0].name}"
            return None
        
        # Handle "the first/second/etc" references
        ordinals = {
            "the first": 0, "the second": 1, "the third": 2,
            "the last": -1, "the previous": -1,
        }
        
        for phrase, index in ordinals.items():
            if phrase in query_lower:
                # Try to resolve from citations
                citations = self.memory.get_recent_citations(5)
                if citations and abs(index) <= len(citations):
                    citation = citations[index]
                    rewritten = query_lower.replace(phrase, f"'{citation.source}'")
                    return rewritten
                
                # Try entities
                entities = self.memory.get_recent_entities(5)
                if entities and abs(index) <= len(entities):
                    entity = entities[index]
                    rewritten = query_lower.replace(phrase, f"'{entity.name}'")
                    return rewritten
        
        return None


class HybridRewriter:
    """
    Combines rule-based and LLM rewriting for efficiency.
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        memory_manager: MemoryManager,
    ):
        self.rule_based = RuleBasedRewriter(memory_manager)
        self.llm_based = QueryRewriter(llm_provider, memory_manager)
    
    def rewrite(self, query: str) -> RewriteResult:
        """
        Rewrite using rules first, then LLM if needed.
        """
        # Try rule-based first
        rule_result = self.rule_based.rewrite(query)
        
        if rule_result:
            return RewriteResult(
                original_query=query,
                rewritten_query=rule_result,
                was_rewritten=True,
                confidence=0.9,
                resolved_references=[],
                reasoning="Resolved using rules",
            )
        
        # Fall back to LLM
        return self.llm_based.rewrite(query)
