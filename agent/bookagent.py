from langchain.agents import create_agent

from .prompts import get_system_prompt

from tools.semantic_search import semantic_search
from tools.web_search import web_search


def build_agent(llm):
    """Build agent with 2 tools: semantic_search (LLM profile routing), web_search (fallback)."""
    # Generate system prompt dynamically to pick up live conversation context & profiles
    system_prompt = get_system_prompt()

    tools = [
        semantic_search,  # LLM profile-based routing + search
        web_search,       # Fallback for external queries
    ]
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
