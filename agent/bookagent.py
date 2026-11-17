from langchain.agents import create_agent

from .prompts import SYSTEM_PROMPT

from tools.semantic_search import semantic_search
from tools.web_search import web_search


def build_agent(llm):
    """Build agent with 2 tools: semantic_search (auto-routing), web_search (fallback)."""
    tools = [
        semantic_search, # Auto-discovers KBs and searches
        web_search,      # Fallback for external queries
    ]
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )