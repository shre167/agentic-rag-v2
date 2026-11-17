"""
Agent Builder
=============
Constructs a ReAct agent with semantic_search and web_search tools.
"""

from langchain.agents import create_agent

from agent.prompts import SYSTEM_PROMPT
from tools.semantic_search import semantic_search
from tools.web_search import web_search


def build_agent(llm):
    """Build ReAct agent with tools."""
    tools = [
        semantic_search,
        web_search,
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
