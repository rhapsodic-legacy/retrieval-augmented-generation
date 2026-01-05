"""
Unified Memory Manager

Coordinates session memory and long-term memory:
- Manages memory lifecycle
- Handles persistence
- Provides unified interface for the RAG system
"""

from typing import Optional
from datetime import datetime
from pathlib import Path

from .session import SessionMemory, ConversationTurn, Citation, Entity
from .long_term import LongTermMemory, UserFact, UserPreference, DocumentMemory


class MemoryManager:
    """
    Unified interface for all memory operations.
    
    Manages:
    - Current session memory
    - Long-term user memory
    - Memory extraction and summarization
    - Persistence across sessions
    """
    
    def __init__(
        self,
        user_id: str,
        storage_path: Optional[str] = None,
        session_id: Optional[str] = None,
        auto_persist: bool = True,
    ):
        """
        Initialize the memory manager.
        
        Args:
            user_id: Unique user identifier
            storage_path: Path for persistent storage
            session_id: Optional session ID (auto-generated if not provided)
            auto_persist: Automatically persist session on end
        """
        self.user_id = user_id
        self.storage_path = Path(storage_path) if storage_path else None
        self.auto_persist = auto_persist
        
        # Initialize memory components
        self.session = SessionMemory(session_id=session_id)
        self.long_term = LongTermMemory(
            user_id=user_id,
            storage_path=storage_path,
        )
        
        # Load session if persisted
        self._load_session_if_exists()
    
    def _load_session_if_exists(self):
        """Load previous session if it exists."""
        if not self.storage_path:
            return
        
        session_file = self.storage_path / f"session_{self.session.session_id}.json"
        if session_file.exists():
            import json
            with open(session_file) as f:
                data = json.load(f)
            self.session = SessionMemory.from_dict(data)
    
    # =========================================================================
    # Session Memory Operations
    # =========================================================================
    
    def add_user_message(
        self,
        content: str,
        original_query: Optional[str] = None,
        rewritten_query: Optional[str] = None,
    ) -> ConversationTurn:
        """Add a user message to current session."""
        return self.session.add_user_message(
            content=content,
            original_query=original_query,
            rewritten_query=rewritten_query,
        )
    
    def add_assistant_message(
        self,
        content: str,
        citations_used: Optional[list[str]] = None,
    ) -> ConversationTurn:
        """Add an assistant message to current session."""
        return self.session.add_assistant_message(
            content=content,
            citations_used=citations_used,
        )
    
    def add_citation(
        self,
        citation_id: str,
        content: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> Citation:
        """Add a citation to current session."""
        return self.session.add_citation(
            citation_id=citation_id,
            content=content,
            source=source,
            metadata=metadata,
        )
    
    def add_entity(
        self,
        name: str,
        entity_type: str,
        first_mention: str,
        attributes: Optional[dict] = None,
    ) -> Entity:
        """Add an entity for pronoun resolution."""
        return self.session.add_entity(
            name=name,
            entity_type=entity_type,
            first_mention=first_mention,
            attributes=attributes,
        )
    
    def get_conversation_context(self, n_turns: int = 10) -> list[dict]:
        """Get recent conversation for context."""
        return self.session.get_messages_for_context(n_turns)
    
    def get_context_string(self) -> str:
        """Get formatted context string for query rewriting."""
        return self.session.build_context_string()
    
    def get_recent_citations(self, n: int = 5) -> list[Citation]:
        """Get recently used citations."""
        return self.session.get_recent_citations(n)
    
    def get_recent_entities(self, n: int = 5) -> list[Entity]:
        """Get recently mentioned entities."""
        return self.session.get_recent_entities(n)
    
    # =========================================================================
    # Long-Term Memory Operations
    # =========================================================================
    
    def remember_fact(
        self,
        fact: str,
        category: str = "general",
        confidence: float = 1.0,
    ) -> UserFact:
        """Remember a fact about the user."""
        return self.long_term.add_fact(
            fact=fact,
            category=category,
            confidence=confidence,
            source_session=self.session.session_id,
        )
    
    def get_user_facts(
        self,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> list[UserFact]:
        """Get remembered facts about the user."""
        return self.long_term.get_facts(category=category, limit=limit)
    
    def search_facts(self, query: str, limit: int = 10) -> list[UserFact]:
        """Search for relevant facts."""
        return self.long_term.search_facts(query, limit)
    
    def set_preference(self, key: str, value, category: str = "general"):
        """Set a user preference."""
        return self.long_term.set_preference(key, value, category)
    
    def get_preference(self, key: str, default=None):
        """Get a user preference."""
        return self.long_term.get_preference(key, default)
    
    def remember_document(
        self,
        doc_id: str,
        title: str,
        summary: Optional[str] = None,
        key_topics: Optional[list[str]] = None,
    ) -> DocumentMemory:
        """Remember information about a document."""
        return self.long_term.remember_document(
            doc_id=doc_id,
            title=title,
            summary=summary,
            key_topics=key_topics,
        )
    
    def get_document_memory(self, doc_id: str) -> Optional[DocumentMemory]:
        """Get memory about a document."""
        return self.long_term.get_document_memory(doc_id)
    
    def get_recent_documents(self, limit: int = 10) -> list[DocumentMemory]:
        """Get recently accessed documents."""
        return self.long_term.get_recent_documents(limit)
    
    # =========================================================================
    # Context Building
    # =========================================================================
    
    def build_full_context(self) -> dict:
        """
        Build complete context for the LLM including:
        - Recent conversation
        - User facts/preferences
        - Active citations
        - Tracked entities
        """
        return {
            "session": self.session.get_context_summary(),
            "recent_messages": self.session.get_messages_for_context(5),
            "recent_citations": [
                {"id": c.id, "source": c.source, "content": c.content[:200]}
                for c in self.session.get_recent_citations(3)
            ],
            "recent_entities": [
                {"name": e.name, "type": e.type}
                for e in self.session.get_recent_entities(5)
            ],
            "user_facts": [
                {"fact": f.fact, "category": f.category}
                for f in self.long_term.get_facts(limit=5)
            ],
            "preferences": self.long_term.get_all_preferences(),
            "recent_documents": [
                {"title": d.title, "doc_id": d.doc_id}
                for d in self.long_term.get_recent_documents(3)
            ],
        }
    
    def build_system_context(self) -> str:
        """Build a system context string for the LLM."""
        parts = []
        
        # User preferences
        prefs = self.long_term.get_all_preferences()
        if prefs:
            parts.append("User preferences:")
            for key, value in prefs.items():
                parts.append(f"  - {key}: {value}")
        
        # Key facts
        facts = self.long_term.get_facts(limit=5)
        if facts:
            parts.append("\nKey facts about user:")
            for fact in facts:
                parts.append(f"  - {fact.fact}")
        
        # Recent documents
        docs = self.long_term.get_recent_documents(3)
        if docs:
            parts.append("\nDocuments user has accessed:")
            for doc in docs:
                parts.append(f"  - {doc.title}")
        
        return "\n".join(parts) if parts else ""
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def start_new_session(self, session_id: Optional[str] = None):
        """Start a new session, optionally persisting the current one."""
        if self.auto_persist:
            self.persist_session()
        
        self.session = SessionMemory(session_id=session_id)
    
    def persist_session(self):
        """Persist current session to storage."""
        if not self.storage_path:
            return
        
        import json
        
        # Save session data
        session_file = self.storage_path / f"session_{self.session.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(self.session.to_dict(), f, indent=2)
        
        # Generate and save session summary
        if self.session.turn_counter > 0:
            summary = self._generate_session_summary()
            topics = self._extract_session_topics()
            
            self.long_term.save_session_summary(
                session_id=self.session.session_id,
                summary=summary,
                topics=topics,
                turn_count=self.session.turn_counter,
            )
    
    def _generate_session_summary(self) -> str:
        """Generate a summary of the current session."""
        turns = self.session.turns
        if not turns:
            return "Empty session"
        
        # Simple summary based on first and last messages
        first_query = turns[0].content if turns else ""
        topics = [e.name for e in self.session.get_recent_entities(3)]
        
        return f"Session about: {first_query[:100]}... Topics: {', '.join(topics) or 'general'}"
    
    def _extract_session_topics(self) -> list[str]:
        """Extract main topics from the session."""
        entities = self.session.get_recent_entities(10)
        return [e.name for e in entities if e.type in ("topic", "concept", "document")]
    
    def end_session(self):
        """End the current session and persist if enabled."""
        if self.auto_persist:
            self.persist_session()
        
        self.session.clear()
    
    # =========================================================================
    # Memory Extraction
    # =========================================================================
    
    def extract_facts_from_conversation(self, llm_provider) -> list[UserFact]:
        """
        Use LLM to extract memorable facts from conversation.
        
        This is called periodically or at session end to populate long-term memory.
        """
        from ..providers.llm import Message
        
        # Build prompt with recent conversation
        messages = self.session.get_messages_for_context(10)
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
        )
        
        prompt = f"""Analyze this conversation and extract key facts about the user that would be useful to remember for future conversations.

Focus on:
- User preferences (how they like things done)
- Domain knowledge (what they know about)
- Personal context (job, interests, goals)
- Document-specific knowledge

Conversation:
{conversation_text}

Return facts as a JSON array:
[
  {{"fact": "User prefers concise answers", "category": "preference"}},
  {{"fact": "User works in healthcare", "category": "context"}}
]

Only return facts that are clearly stated or strongly implied. Return empty array if no facts found."""

        response = llm_provider.generate(
            messages=[Message(role="user", content=prompt)],
            max_tokens=500,
        )
        
        # Parse facts
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response.content)
            if json_match:
                facts_data = json.loads(json_match.group())
                
                extracted = []
                for fact_data in facts_data:
                    fact = self.remember_fact(
                        fact=fact_data.get("fact", ""),
                        category=fact_data.get("category", "general"),
                    )
                    extracted.append(fact)
                
                return extracted
        except json.JSONDecodeError:
            pass
        
        return []
    
    # =========================================================================
    # Stats and Export
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "session": {
                "session_id": self.session.session_id,
                "turns": self.session.turn_counter,
                "citations": len(self.session.citations),
                "entities": len(self.session.entities),
            },
            "long_term": {
                "facts": len(self.long_term.get_facts(limit=1000)),
                "documents": len(self.long_term.get_recent_documents(1000)),
                "sessions": len(self.long_term.get_session_summaries(1000)),
            },
        }
    
    def export_all(self) -> dict:
        """Export all memory data."""
        return {
            "user_id": self.user_id,
            "session": self.session.to_dict(),
            "long_term": self.long_term.export_all(),
        }
