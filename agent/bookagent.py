"""
Agent Builder
=============
Constructs an autonomous ReAct agent with two core tools:
  1. semantic_search — Dynamic KB discovery + embedding/LLM semantic router + multi-book retrieval
  2. web_search      — DuckDuckGo search for live/external topics or when books have NO_MATCH
"""

from langchain.agents import create_agent

from agent.prompts import SYSTEM_PROMPT
from tools.semantic_search import semantic_search
from tools.web_search import web_search


def build_agent(llm):
    """Build ReAct agent with semantic_search and web_search tools."""
    tools = [
        semantic_search,  # Automatically routes to relevant KB(s) and retrieves chunks
        web_search,       # Live web search for non-book or external topics
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
