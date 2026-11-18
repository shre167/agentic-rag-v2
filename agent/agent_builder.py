"""
Agent Builder
=============
Constructs a ReAct agent with semantic_search and web_search tools.
Uses dynamically generated system prompt with auto-discovered book profiles.
"""

from langchain.agents import create_agent

from agent.prompts import get_system_prompt
from tools.semantic_search import semantic_search
from tools.web_search import web_search


def build_agent(llm):
    """Build ReAct agent with tools and dynamic system prompt."""
    # Generate system prompt dynamically to pick up live conversation context & profiles
    system_prompt = get_system_prompt()

    tools = [
        semantic_search,
        web_search,
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
