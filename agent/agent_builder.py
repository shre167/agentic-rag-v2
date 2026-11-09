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
from tools.list_books import list_books
from tools.semantic_search import semantic_search


def build_agent(llm):
    """
    Build ReAct agent with tools.

    Tools:
      1. list_books: Discover which KBs exist
      2. semantic_search: Metadata-aware semantic search (handles page/chapter/section)

    Args:
        llm: LLM instance

    Returns:
        Agent: ReAct agent ready to answer questions
    """
    tools = [
        list_books,       # Discover available books - agent calls this FIRST
        semantic_search,  # General semantic search - for concepts, chapters, sections, pages
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent
