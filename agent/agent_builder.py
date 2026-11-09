"""
Agent Builder

Constructs a ReAct agent with available tools.

Pipeline:
  1. Load tool definitions
  2. Create system prompt
  3. Create ReAct agent
  4. Return agent instance

The agent uses dynamic decision-making to choose tools and parameters.
No hardcoded logic - all decisions made by LLM.
"""

from langchain.agents import create_agent

from agent.prompts import SYSTEM_PROMPT
from agent.tools.list_books import list_books
from agent.tools.semantic_search import semantic_search
from agent.tools.page_search import page_search


def build_agent(llm):
    """
    Build ReAct agent with tools.
    
    Why three tools?
      1. list_books: Agent needs to discover which KBs exist
      2. semantic_search: General semantic search with optional filters
      3. page_search: Specialized for page queries (faster, simpler)
    
    Why no web_search by default?
      - User said: only with explicit approval
      - Set up via hook in .kiro/hooks/approve-web-search.json
      - Hook intercepts web_search calls and asks for confirmation
    
    Args:
        llm: LLM instance from config/llm_config.py
    
    Returns:
        Agent: ReAct agent ready to answer questions
    
    Pipeline:
      1. Define tools list (what agent can do)
      2. Set system prompt (how agent should behave)
      3. Create agent with create_agent()
      4. Return instance
    """
    
    # Define tools: order matters - list_books should be first
    tools = [
        list_books,       # Discover available books - agent calls this FIRST
        page_search,      # Page-specific search - fast for "what's on page X"
        semantic_search,  # General semantic search - for concepts, chapters, sections
    ]
    
    # Create agent
    # ReAct = Reasoning + Acting
    # Agent alternates: think (decide tool) -> act (call tool) -> think (process result)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    
    return agent
