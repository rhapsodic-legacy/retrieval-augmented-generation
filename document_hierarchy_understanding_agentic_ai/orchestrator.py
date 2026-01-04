"""
Group chat orchestration for multi-agent collaboration.

This module sets up the group chat where agents collaborate
to answer complex queries about hierarchical documents.
"""

from typing import Optional, Callable
import autogen
from autogen import GroupChat, GroupChatManager

from .agents import AgentFactory, get_llm_config


class HierarchicalRAGGroupChat:
    """
    Orchestrates multi-agent collaboration for hierarchical RAG.
    
    The group chat enables agents to:
    - Discuss query interpretation
    - Share retrieved information
    - Verify and refine answers
    - Produce comprehensive, cited responses
    """
    
    def __init__(
        self,
        tool_functions: dict[str, Callable],
        model: str = "claude-sonnet-4-20250514",
        max_rounds: int = 15,
        verbose: bool = True
    ):
        """
        Initialize the group chat.
        
        Args:
            tool_functions: Dictionary of tool functions for agents
            model: LLM model to use
            max_rounds: Maximum conversation rounds
            verbose: Print conversation to console
        """
        self.verbose = verbose
        self.max_rounds = max_rounds
        
        # Create agents
        factory = AgentFactory(model, tool_functions)
        self.agents = factory.create_all_agents()
        
        # Extract agents
        self.coordinator = self.agents["coordinator"]
        self.retriever = self.agents["retriever"]
        self.crossref = self.agents["crossref"]
        self.summarizer = self.agents["summarizer"]
        self.verifier = self.agents["verifier"]
        self.user_proxy = self.agents["user_proxy"]
        
        # Create group chat
        self.group_chat = GroupChat(
            agents=[
                self.user_proxy,
                self.coordinator,
                self.retriever,
                self.crossref,
                self.summarizer,
                self.verifier,
            ],
            messages=[],
            max_round=max_rounds,
            speaker_selection_method=self._select_speaker,
            allow_repeat_speaker=False,
        )
        
        # Create manager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=get_llm_config(model),
        )
    
    def _select_speaker(
        self,
        last_speaker,
        group_chat: GroupChat
    ):
        """
        Custom speaker selection logic.
        
        Flow:
        1. User → Coordinator (understand query)
        2. Coordinator → Retriever (search)
        3. Retriever → CrossRef (check references)
        4. CrossRef → Summarizer (synthesize)
        5. Summarizer → Verifier (verify)
        6. Verifier → Coordinator (final answer)
        """
        messages = group_chat.messages
        
        if not messages:
            return self.coordinator
        
        last_message = messages[-1].get("content", "")
        
        # Based on last speaker, determine next
        if last_speaker == self.user_proxy:
            return self.coordinator
        
        elif last_speaker == self.coordinator:
            # Check what coordinator is asking for
            if "search" in last_message.lower() or "retrieve" in last_message.lower():
                return self.retriever
            elif "reference" in last_message.lower() or "citation" in last_message.lower():
                return self.crossref
            elif "summarize" in last_message.lower() or "synthesize" in last_message.lower():
                return self.summarizer
            elif "verify" in last_message.lower() or "check" in last_message.lower():
                return self.verifier
            else:
                return self.retriever  # Default to retrieval
        
        elif last_speaker == self.retriever:
            # After retrieval, check for cross-references
            if "reference" in last_message.lower() or "see section" in last_message.lower():
                return self.crossref
            else:
                return self.summarizer
        
        elif last_speaker == self.crossref:
            return self.summarizer
        
        elif last_speaker == self.summarizer:
            return self.verifier
        
        elif last_speaker == self.verifier:
            # Verification done, back to coordinator for final answer
            return self.coordinator
        
        # Default
        return self.coordinator
    
    def query(self, question: str) -> str:
        """
        Run a query through the multi-agent system.
        
        Args:
            question: The user's question
            
        Returns:
            The final answer from the agent collaboration
        """
        # Reset conversation
        self.group_chat.messages = []
        
        # Start the conversation
        self.user_proxy.initiate_chat(
            self.manager,
            message=question,
            clear_history=True,
        )
        
        # Extract final answer from conversation
        return self._extract_answer()
    
    def _extract_answer(self) -> str:
        """Extract the final answer from the conversation."""
        messages = self.group_chat.messages
        
        # Look for the last coordinator message (usually the final answer)
        for msg in reversed(messages):
            if msg.get("name") == "Coordinator":
                content = msg.get("content", "")
                if content and "TERMINATE" not in content:
                    return content
        
        # Fallback: return last non-empty message
        for msg in reversed(messages):
            content = msg.get("content", "")
            if content and "TERMINATE" not in content:
                return content
        
        return "No answer generated."
    
    def get_conversation_history(self) -> list[dict]:
        """Get the full conversation history."""
        return self.group_chat.messages


class SequentialRAGWorkflow:
    """
    A simpler sequential workflow (not group chat).
    
    Agents are called in a fixed sequence:
    User → Coordinator → Retriever → Summarizer → Verifier → Answer
    
    Easier to debug and more predictable than group chat.
    """
    
    def __init__(
        self,
        tool_functions: dict[str, Callable],
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True
    ):
        self.verbose = verbose
        
        # Create agents
        factory = AgentFactory(model, tool_functions)
        self.agents = factory.create_all_agents()
        
        self.coordinator = self.agents["coordinator"]
        self.retriever = self.agents["retriever"]
        self.summarizer = self.agents["summarizer"]
        self.verifier = self.agents["verifier"]
        self.user_proxy = self.agents["user_proxy"]
    
    def query(self, question: str) -> dict:
        """
        Run a query through the sequential workflow.
        
        Returns:
            Dictionary with answer and intermediate results
        """
        results = {
            "question": question,
            "steps": [],
            "answer": "",
        }
        
        # Step 1: Coordinator analyzes the query
        if self.verbose:
            print("\n🎯 Step 1: Coordinator analyzing query...")
        
        coord_response = self.coordinator.generate_reply(
            messages=[{
                "role": "user",
                "content": f"Analyze this query and determine the best retrieval strategy: {question}"
            }]
        )
        results["steps"].append({
            "agent": "Coordinator",
            "action": "Query Analysis",
            "output": coord_response
        })
        
        # Step 2: Retriever searches
        if self.verbose:
            print("🔍 Step 2: Retriever searching...")
        
        retriever_response = self.retriever.generate_reply(
            messages=[{
                "role": "user", 
                "content": f"Search for content relevant to: {question}\n\nCoordinator guidance: {coord_response}"
            }]
        )
        results["steps"].append({
            "agent": "Retriever",
            "action": "Content Search",
            "output": retriever_response
        })
        
        # Step 3: Summarizer synthesizes
        if self.verbose:
            print("📝 Step 3: Summarizer synthesizing...")
        
        summarizer_response = self.summarizer.generate_reply(
            messages=[{
                "role": "user",
                "content": f"""Synthesize an answer to: {question}
                
Retrieved content:
{retriever_response}

Create a comprehensive answer with citations."""
            }]
        )
        results["steps"].append({
            "agent": "Summarizer",
            "action": "Synthesis",
            "output": summarizer_response
        })
        
        # Step 4: Verifier checks
        if self.verbose:
            print("✅ Step 4: Verifier checking...")
        
        verifier_response = self.verifier.generate_reply(
            messages=[{
                "role": "user",
                "content": f"""Verify this answer:

Question: {question}

Answer: {summarizer_response}

Retrieved content: {retriever_response}

Check for accuracy and proper citations."""
            }]
        )
        results["steps"].append({
            "agent": "Verifier",
            "action": "Verification",
            "output": verifier_response
        })
        
        # Final answer
        results["answer"] = summarizer_response
        results["verification"] = verifier_response
        
        return results


class TwoAgentRAG:
    """
    Minimal two-agent setup for simpler use cases.
    
    Just a retriever agent that searches and answers,
    with a user proxy to execute tools.
    """
    
    SYSTEM_PROMPT = """You are a Hierarchical Document RAG assistant.

Your job is to:
1. Search for relevant content using your tools
2. Include hierarchical context (document > section > paragraph)
3. Provide well-cited answers

For each query:
1. First search summaries to find relevant documents/sections
2. Then search specific content
3. Get parent context for matched paragraphs
4. Synthesize a complete answer with citations

Always cite your sources with the document/section structure.

When you have a complete answer, end with TERMINATE."""

    def __init__(
        self,
        tool_functions: dict[str, Callable],
        model: str = "claude-sonnet-4-20250514",
    ):
        self.llm_config = get_llm_config(model)
        
        # Create assistant
        self.assistant = autogen.AssistantAgent(
            name="RAGAssistant",
            system_message=self.SYSTEM_PROMPT,
            llm_config=self.llm_config,
        )
        
        # Register tools
        for name, func in tool_functions.items():
            self.assistant.register_for_llm(name=name, description=func.__doc__)(func)
        
        # Create user proxy
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )
        
        # Register tool execution
        for name, func in tool_functions.items():
            self.user_proxy.register_for_execution(name=name)(func)
    
    def query(self, question: str) -> str:
        """Run a query and return the answer."""
        self.user_proxy.initiate_chat(
            self.assistant,
            message=question,
            clear_history=True,
        )
        
        # Get last assistant message
        for msg in reversed(self.user_proxy.chat_messages[self.assistant]):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                return content.replace("TERMINATE", "").strip()
        
        return "No answer generated."
