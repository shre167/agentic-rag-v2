from langchain.agents import create_agent

from .prompts import SYSTEM_PROMPT

from tools.semantic_search import semantic_search
from tools.list_books import list_books
from tools.web_search import web_search


def build_agent(llm):
    """Build agent with 3 tools: list_books, semantic_search (primary), web_search (fallback)."""
    tools = [
        list_books,      # Discover available books
        semantic_search, # Search book content (PRIMARY)
        web_search,      # Search web (FALLBACK ONLY - for irrelevant queries)
    ]
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )