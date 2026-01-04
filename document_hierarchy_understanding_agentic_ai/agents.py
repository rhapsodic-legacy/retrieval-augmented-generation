"""
Agent definitions for Hierarchical Document RAG.

This module defines specialized agents that collaborate to:
1. Parse and understand document structure
2. Retrieve relevant content with context
3. Detect and follow cross-references
4. Synthesize and verify answers
"""

from typing import Optional, Callable
import autogen
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent


# =============================================================================
# AGENT CONFIGURATIONS
# =============================================================================

def get_llm_config(model: str = "claude-sonnet-4-20250514") -> dict:
    """Get the LLM configuration for agents."""
    return {
        "config_list": [
            {
                "model": model,
                "api_type": "anthropic",
                # API key from environment variable ANTHROPIC_API_KEY
            }
        ],
        "temperature": 0.1,
        "cache_seed": None,  # Disable caching for dynamic responses
    }


# =============================================================================
# COORDINATOR AGENT
# =============================================================================

COORDINATOR_SYSTEM_PROMPT = """You are the Research Coordinator for a hierarchical document RAG system.

Your role is to:
1. Understand user queries and break them into subtasks
2. Delegate to specialized agents (Retriever, CrossReference, Summarizer, Verifier)
3. Synthesize their responses into a coherent answer
4. Ensure answers include proper citations and context

When a user asks a question:
1. First, determine if it's a BROAD query (overview, general) or SPECIFIC query (detailed, exact)
2. For BROAD queries: Start with summary search, then drill down
3. For SPECIFIC queries: Search content directly, expand context
4. Always verify cross-references when documents mention other sections

Coordinate the agents by requesting their specific capabilities:
- Ask RETRIEVER to search for relevant content
- Ask CROSSREF to find and resolve references
- Ask SUMMARIZER to synthesize information
- Ask VERIFIER to check accuracy and add citations

Always aim for comprehensive, well-cited answers that respect document hierarchy."""


def create_coordinator_agent(llm_config: dict) -> AssistantAgent:
    """Create the coordinator agent that orchestrates the workflow."""
    return AssistantAgent(
        name="Coordinator",
        system_message=COORDINATOR_SYSTEM_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )


# =============================================================================
# RETRIEVER AGENT
# =============================================================================

RETRIEVER_SYSTEM_PROMPT = """You are the Retriever Agent for a hierarchical document RAG system.

Your specialized capabilities:
1. Search the vector store for relevant content
2. Expand context by retrieving parent sections/documents
3. Find sibling paragraphs for surrounding context
4. Use summary-first search for broad queries

When asked to retrieve information:
1. Use search_content() for specific queries
2. Use search_summaries() first for broad queries to find relevant documents
3. Always use get_parent_context() to include hierarchical context
4. Use get_siblings() when surrounding context would help

Report your findings with:
- The matched content
- The hierarchical context (Document > Section > Subsection)
- Relevance scores
- Suggestions for what to search next if needed

You have access to these tools:
- search_content: Vector search for content
- search_summaries: Search document/section summaries
- get_node_content: Get full content of a node
- get_parent_context: Get parent chain
- get_siblings: Get adjacent nodes"""


def create_retriever_agent(
    llm_config: dict,
    tool_functions: dict[str, Callable]
) -> AssistantAgent:
    """Create the retriever agent with search tools."""
    agent = AssistantAgent(
        name="Retriever",
        system_message=RETRIEVER_SYSTEM_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    # Register tools
    for name, func in tool_functions.items():
        if name in ["search_content", "search_summaries", "get_node_content", 
                    "get_parent_context", "get_siblings"]:
            agent.register_for_llm(name=name, description=func.__doc__)(func)
    
    return agent


# =============================================================================
# CROSS-REFERENCE AGENT
# =============================================================================

CROSSREF_SYSTEM_PROMPT = """You are the Cross-Reference Agent for a hierarchical document RAG system.

Your specialized capabilities:
1. Detect references in content (e.g., "See Section 3.2", citations [1])
2. Resolve references to their target nodes
3. Find related content through citation networks
4. Identify when information from multiple documents is needed

When asked to analyze references:
1. Use find_references_in_node() to detect all references in content
2. Use get_related_nodes() to follow reference chains
3. Report which references are resolved vs unresolved
4. Suggest additional content that should be retrieved

Your analysis helps ensure:
- Cross-referenced content is included in answers
- Citations are properly tracked
- Related documents are considered

You have access to these tools:
- find_references_in_node: Find all references in a node
- get_related_nodes: Follow reference chains
- get_node_content: Get content of referenced nodes"""


def create_crossref_agent(
    llm_config: dict,
    tool_functions: dict[str, Callable]
) -> AssistantAgent:
    """Create the cross-reference agent."""
    agent = AssistantAgent(
        name="CrossReference",
        system_message=CROSSREF_SYSTEM_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    # Register tools
    for name, func in tool_functions.items():
        if name in ["find_references_in_node", "get_related_nodes", "get_node_content"]:
            agent.register_for_llm(name=name, description=func.__doc__)(func)
    
    return agent


# =============================================================================
# SUMMARIZER AGENT
# =============================================================================

SUMMARIZER_SYSTEM_PROMPT = """You are the Summarizer Agent for a hierarchical document RAG system.

Your specialized capabilities:
1. Synthesize information from multiple retrieved passages
2. Create query-focused summaries that directly answer questions
3. Preserve hierarchical context in summaries
4. Handle conflicting information across sources

When asked to summarize:
1. Focus on information relevant to the original query
2. Maintain the document hierarchy in your summary (cite sections)
3. Note any contradictions or ambiguities
4. Highlight the most authoritative sources

Your summaries should:
- Directly answer the user's question
- Reference the document structure (e.g., "According to Section 3...")
- Be concise but complete
- Note when information is uncertain or conflicting

Output format:
1. Main answer (2-3 sentences)
2. Supporting details (bullet points if needed)
3. Source references (document/section names)
4. Any caveats or limitations"""


def create_summarizer_agent(llm_config: dict) -> AssistantAgent:
    """Create the summarizer agent."""
    return AssistantAgent(
        name="Summarizer",
        system_message=SUMMARIZER_SYSTEM_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )


# =============================================================================
# VERIFIER AGENT
# =============================================================================

VERIFIER_SYSTEM_PROMPT = """You are the Verifier Agent for a hierarchical document RAG system.

Your specialized capabilities:
1. Verify that answers are supported by retrieved content
2. Check for factual consistency across sources
3. Ensure proper citations are included
4. Identify gaps or unsupported claims

When verifying an answer:
1. Cross-check each claim against the source content
2. Verify citations point to correct sources
3. Identify any claims not supported by the retrieved content
4. Check for contradictions between sources

Your verification report should include:
- ✅ Verified claims (with source)
- ⚠️ Partially supported claims
- ❌ Unsupported claims
- 📚 Suggested additional sources to check

Quality criteria:
- All factual claims must have source support
- Citations must be accurate
- Hierarchical context should be preserved
- Uncertainty should be acknowledged

You have access to these tools for verification:
- get_node_content: Verify content matches citations
- search_content: Find additional supporting evidence"""


def create_verifier_agent(
    llm_config: dict,
    tool_functions: dict[str, Callable]
) -> AssistantAgent:
    """Create the verifier agent."""
    agent = AssistantAgent(
        name="Verifier",
        system_message=VERIFIER_SYSTEM_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    
    # Register tools
    for name, func in tool_functions.items():
        if name in ["get_node_content", "search_content"]:
            agent.register_for_llm(name=name, description=func.__doc__)(func)
    
    return agent


# =============================================================================
# USER PROXY AGENT
# =============================================================================

def create_user_proxy(
    tool_functions: dict[str, Callable],
    human_input_mode: str = "NEVER"
) -> UserProxyAgent:
    """
    Create a user proxy that can execute tools.
    
    The user proxy executes tool calls on behalf of the agents.
    """
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,  # Don't execute arbitrary code
    )
    
    # Register tool execution
    for name, func in tool_functions.items():
        user_proxy.register_for_execution(name=name)(func)
    
    return user_proxy


# =============================================================================
# AGENT FACTORY
# =============================================================================

class AgentFactory:
    """Factory for creating the agent ensemble."""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        tool_functions: Optional[dict[str, Callable]] = None
    ):
        self.llm_config = get_llm_config(model)
        self.tool_functions = tool_functions or {}
    
    def create_all_agents(self) -> dict[str, ConversableAgent]:
        """Create all agents for the system."""
        return {
            "coordinator": create_coordinator_agent(self.llm_config),
            "retriever": create_retriever_agent(self.llm_config, self.tool_functions),
            "crossref": create_crossref_agent(self.llm_config, self.tool_functions),
            "summarizer": create_summarizer_agent(self.llm_config),
            "verifier": create_verifier_agent(self.llm_config, self.tool_functions),
            "user_proxy": create_user_proxy(self.tool_functions),
        }
    
    def create_minimal_agents(self) -> dict[str, ConversableAgent]:
        """Create a minimal set of agents (coordinator + retriever)."""
        return {
            "coordinator": create_coordinator_agent(self.llm_config),
            "retriever": create_retriever_agent(self.llm_config, self.tool_functions),
            "user_proxy": create_user_proxy(self.tool_functions),
        }
