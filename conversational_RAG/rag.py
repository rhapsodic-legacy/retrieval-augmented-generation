"""
Conversational RAG with Memory

Main orchestrator that combines:
- Multi-provider LLM support (Claude, Gemini)
- Session and long-term memory
- Conversation-aware query rewriting
- Citation tracking across turns
"""

from dataclasses import dataclass, field
from typing import Optional, Generator
from pathlib import Path

from .providers.llm import (
    BaseLLMProvider,
    Message,
    create_provider,
)
from .providers.embeddings import (
    BaseEmbeddingProvider,
    create_embedding_provider,
)
from .memory.manager import MemoryManager
from .memory.session import Citation
from .query_rewriter import QueryRewriter, HybridRewriter, RewriteResult
from .citation_tracker import CitationTracker, ConversationCitationManager
from .core.retriever import DocumentStore, ChunkedDocumentStore, RetrievalResult


@dataclass
class ConversationConfig:
    """Configuration for conversational RAG."""
    
    # LLM settings
    llm_provider: str = "anthropic"  # "anthropic" or "gemini"
    llm_model: Optional[str] = None
    
    # Embedding settings
    embedding_provider: str = "local"  # "voyage", "google", "openai", "local"
    embedding_model: Optional[str] = None
    
    # Memory settings
    user_id: str = "default_user"
    storage_path: Optional[str] = None
    auto_persist: bool = True
    
    # Retrieval settings
    n_results: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Query rewriting
    enable_rewriting: bool = True
    rewrite_mode: str = "hybrid"  # "hybrid", "llm", "rules"
    
    # Citation settings
    citation_format: str = "bracket"  # "bracket", "superscript", "footnote"
    
    # Generation settings
    max_tokens: int = 1000
    temperature: float = 0.1
    
    # Memory extraction
    extract_facts_every_n_turns: int = 10


@dataclass
class ConversationResponse:
    """Response from a conversation turn."""
    answer: str
    citations: list[str]  # Citation display IDs
    sources_used: list[dict]
    
    # Query processing info
    original_query: str
    rewritten_query: Optional[str]
    was_rewritten: bool
    
    # Memory info
    entities_mentioned: list[str]
    facts_extracted: list[str]
    
    # Metadata
    turn_number: int
    metadata: dict = field(default_factory=dict)


class ConversationalRAG:
    """
    Production-grade conversational RAG with memory.
    
    Features:
    - Multi-turn conversation with context awareness
    - Query rewriting for pronouns and references
    - Session memory (current conversation)
    - Long-term memory (persisted across sessions)
    - Citation tracking across turns
    - Support for Claude and Gemini
    
    Usage:
        rag = ConversationalRAG(config)
        rag.add_document("path/to/doc.md")
        
        response = rag.chat("What is this document about?")
        print(response.answer)
        
        response = rag.chat("Tell me more about the second point")
        print(response.answer)  # Properly resolves "second point"
    """
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided documents.

{user_context}

Guidelines:
- Answer based on the provided context
- If the context doesn't contain the answer, say so
- Be concise but complete
- Reference specific sections when relevant
- Maintain conversation continuity

{memory_context}"""

    ANSWER_PROMPT = """Based on the following context, answer the user's question.

RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Provide a helpful, accurate answer based on the context. If citing specific information, note where it came from."""

    def __init__(self, config: Optional[ConversationConfig] = None):
        """
        Initialize conversational RAG.
        
        Args:
            config: Configuration options
        """
        self.config = config or ConversationConfig()
        
        # Initialize providers
        self.llm = create_provider(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        )
        
        self.embeddings = create_embedding_provider(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
        )
        
        # Initialize memory
        self.memory = MemoryManager(
            user_id=self.config.user_id,
            storage_path=self.config.storage_path,
            auto_persist=self.config.auto_persist,
        )
        
        # Initialize document store
        self.documents = ChunkedDocumentStore(
            embedding_provider=self.embeddings,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            persist_directory=self.config.storage_path,
        )
        
        # Initialize query rewriter
        if self.config.enable_rewriting:
            if self.config.rewrite_mode == "hybrid":
                self.rewriter = HybridRewriter(self.llm, self.memory)
            else:
                self.rewriter = QueryRewriter(self.llm, self.memory)
        else:
            self.rewriter = None
        
        # Initialize citation tracker
        self.citations = CitationTracker(
            citation_format=self.config.citation_format
        )
        self.citation_manager = ConversationCitationManager(self.citations)
        
        # Track turns
        self.turn_count = 0
    
    # =========================================================================
    # Document Management
    # =========================================================================
    
    def add_document(
        self,
        content: str,
        title: str = "",
        source: str = "",
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document text
            title: Document title
            source: Source path/URL
            doc_id: Optional ID
            
        Returns:
            Document ID
        """
        # Add to document store
        chunk_ids = self.documents.add_document(
            content=content,
            doc_id=doc_id,
            title=title,
            source=source,
        )
        
        # Remember in long-term memory
        self.memory.remember_document(
            doc_id=chunk_ids[0].split("_chunk_")[0] if chunk_ids else doc_id or "unknown",
            title=title,
            summary=content[:200] + "..." if len(content) > 200 else content,
        )
        
        return chunk_ids[0] if chunk_ids else ""
    
    def add_document_from_file(self, file_path: str) -> str:
        """Add a document from a file."""
        path = Path(file_path)
        content = path.read_text(encoding='utf-8')
        
        return self.add_document(
            content=content,
            title=path.stem,
            source=str(path),
        )
    
    # =========================================================================
    # Chat Interface
    # =========================================================================
    
    def chat(self, message: str) -> ConversationResponse:
        """
        Process a chat message and return a response.
        
        Handles:
        - Query rewriting (pronouns, references)
        - Retrieval
        - Answer generation
        - Citation tracking
        - Memory updates
        """
        self.turn_count += 1
        self.citations.new_turn()
        
        # Check if this is a source query
        source_response = self.citation_manager.handle_source_query(message)
        if source_response:
            self.memory.add_user_message(message)
            self.memory.add_assistant_message(source_response)
            
            return ConversationResponse(
                answer=source_response,
                citations=[],
                sources_used=[],
                original_query=message,
                rewritten_query=None,
                was_rewritten=False,
                entities_mentioned=[],
                facts_extracted=[],
                turn_number=self.turn_count,
            )
        
        # Rewrite query if needed
        rewrite_result = None
        query_to_use = message
        
        if self.rewriter:
            rewrite_result = self.rewriter.rewrite(message)
            if rewrite_result.was_rewritten:
                query_to_use = rewrite_result.rewritten_query
        
        # Retrieve relevant content
        retrieval_results = self.documents.search(
            query=query_to_use,
            n_results=self.config.n_results,
        )
        
        # Add citations
        citation_ids = self.citation_manager.cite_retrieval_results(
            results=[
                {
                    "content": r.content,
                    "source": r.source,
                    "id": r.id,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in retrieval_results
            ],
            query=query_to_use,
        )
        
        # Build context
        context = self._build_retrieval_context(retrieval_results, citation_ids)
        history = self._build_conversation_history()
        
        # Generate answer
        answer = self._generate_answer(
            question=query_to_use,
            context=context,
            history=history,
        )
        
        # Add citation list to answer
        answer_with_citations = answer + self.citations.build_citation_list()
        
        # Update memory
        self.memory.add_user_message(
            content=message,
            original_query=message,
            rewritten_query=query_to_use if rewrite_result and rewrite_result.was_rewritten else None,
        )
        self.memory.add_assistant_message(
            content=answer,
            citations_used=citation_ids,
        )
        
        # Extract entities from the exchange
        entities = self._extract_entities(message, answer, retrieval_results)
        
        # Periodically extract facts for long-term memory
        facts_extracted = []
        if self.turn_count % self.config.extract_facts_every_n_turns == 0:
            facts = self.memory.extract_facts_from_conversation(self.llm)
            facts_extracted = [f.fact for f in facts]
        
        return ConversationResponse(
            answer=answer_with_citations,
            citations=citation_ids,
            sources_used=[
                {
                    "id": r.id,
                    "source": r.source,
                    "score": r.score,
                    "preview": r.content[:100],
                }
                for r in retrieval_results
            ],
            original_query=message,
            rewritten_query=query_to_use if rewrite_result and rewrite_result.was_rewritten else None,
            was_rewritten=rewrite_result.was_rewritten if rewrite_result else False,
            entities_mentioned=[e.name for e in entities],
            facts_extracted=facts_extracted,
            turn_number=self.turn_count,
        )
    
    def chat_stream(self, message: str) -> Generator[str, None, ConversationResponse]:
        """
        Stream a chat response.
        
        Yields chunks of the answer, then returns full ConversationResponse.
        """
        self.turn_count += 1
        self.citations.new_turn()
        
        # Rewrite query
        rewrite_result = None
        query_to_use = message
        
        if self.rewriter:
            rewrite_result = self.rewriter.rewrite(message)
            if rewrite_result.was_rewritten:
                query_to_use = rewrite_result.rewritten_query
        
        # Retrieve
        retrieval_results = self.documents.search(
            query=query_to_use,
            n_results=self.config.n_results,
        )
        
        # Add citations
        citation_ids = self.citation_manager.cite_retrieval_results(
            results=[
                {
                    "content": r.content,
                    "source": r.source,
                    "id": r.id,
                    "score": r.score,
                }
                for r in retrieval_results
            ],
            query=query_to_use,
        )
        
        # Build context
        context = self._build_retrieval_context(retrieval_results, citation_ids)
        history = self._build_conversation_history()
        
        # Stream answer
        prompt = self.ANSWER_PROMPT.format(
            context=context,
            history=history,
            question=query_to_use,
        )
        
        system = self._build_system_prompt()
        full_answer = ""
        
        for chunk in self.llm.generate_stream(
            messages=[Message(role="user", content=prompt)],
            system_prompt=system,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ):
            full_answer += chunk
            yield chunk
        
        # Yield citation list
        citation_list = self.citations.build_citation_list()
        yield citation_list
        full_answer += citation_list
        
        # Update memory
        self.memory.add_user_message(
            content=message,
            original_query=message,
            rewritten_query=query_to_use if rewrite_result and rewrite_result.was_rewritten else None,
        )
        self.memory.add_assistant_message(
            content=full_answer,
            citations_used=citation_ids,
        )
        
        # Return full response
        return ConversationResponse(
            answer=full_answer,
            citations=citation_ids,
            sources_used=[
                {"id": r.id, "source": r.source, "score": r.score}
                for r in retrieval_results
            ],
            original_query=message,
            rewritten_query=query_to_use if rewrite_result and rewrite_result.was_rewritten else None,
            was_rewritten=rewrite_result.was_rewritten if rewrite_result else False,
            entities_mentioned=[],
            facts_extracted=[],
            turn_number=self.turn_count,
        )
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with user context."""
        user_context = self.memory.build_system_context()
        memory_context = ""
        
        # Add recent document context
        docs = self.memory.get_recent_documents(3)
        if docs:
            memory_context = "User has recently accessed: " + ", ".join(d.title for d in docs)
        
        return self.SYSTEM_PROMPT.format(
            user_context=user_context or "No user preferences set.",
            memory_context=memory_context,
        )
    
    def _build_retrieval_context(
        self,
        results: list[RetrievalResult],
        citation_ids: list[str],
    ) -> str:
        """Build context string from retrieval results."""
        parts = []
        
        for result, cite_id in zip(results, citation_ids):
            parts.append(f"{cite_id} From \"{result.source}\":")
            parts.append(result.content)
            parts.append("")
        
        return "\n".join(parts)
    
    def _build_conversation_history(self, n_turns: int = 5) -> str:
        """Build conversation history string."""
        messages = self.memory.get_conversation_context(n_turns)
        
        if not messages:
            return "No previous conversation."
        
        parts = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:300]
            if len(msg["content"]) > 300:
                content += "..."
            parts.append(f"{role}: {content}")
        
        return "\n".join(parts)
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        history: str,
    ) -> str:
        """Generate an answer using the LLM."""
        prompt = self.ANSWER_PROMPT.format(
            context=context,
            history=history,
            question=question,
        )
        
        system = self._build_system_prompt()
        
        response = self.llm.generate(
            messages=[Message(role="user", content=prompt)],
            system_prompt=system,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        return response.content
    
    def _extract_entities(
        self,
        query: str,
        answer: str,
        results: list[RetrievalResult],
    ) -> list:
        """Extract entities from the exchange for memory."""
        entities = []
        
        # Add documents as entities
        for result in results:
            entity = self.memory.add_entity(
                name=result.source,
                entity_type="document",
                first_mention=f"Retrieved for: {query[:50]}",
            )
            entities.append(entity)
        
        return entities
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def new_session(self, session_id: Optional[str] = None):
        """Start a new conversation session."""
        self.memory.start_new_session(session_id)
        self.citations.reset()
        self.turn_count = 0
    
    def end_session(self):
        """End the current session and persist memory."""
        self.memory.end_session()
    
    # =========================================================================
    # Memory Management
    # =========================================================================
    
    def remember(self, fact: str, category: str = "general"):
        """Explicitly remember a fact about the user."""
        return self.memory.remember_fact(fact, category)
    
    def set_preference(self, key: str, value):
        """Set a user preference."""
        return self.memory.set_preference(key, value)
    
    def get_preference(self, key: str, default=None):
        """Get a user preference."""
        return self.memory.get_preference(key, default)
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        return self.memory.get_stats()
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get overall statistics."""
        return {
            "turn_count": self.turn_count,
            "documents": self.documents.get_stats(),
            "memory": self.memory.get_stats(),
            "citations": self.citations.get_stats(),
            "llm_provider": self.config.llm_provider,
            "embedding_provider": self.config.embedding_provider,
        }
