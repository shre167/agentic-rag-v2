from langchain.agents import create_agent

from .prompts import SYSTEM_PROMPT

from tools.semantic_search import semantic_search
from tools.list_books import list_books
from tools.web_search import web_search


def build_agent(llm):
    """Build agent with 3 tools: list_books, semantic_search, web_search."""
    tools = [
        list_books,      # Discover available books
        semantic_search, # Search book content
        web_search,      # Search web (for external info)
    ]
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )