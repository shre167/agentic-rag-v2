"""Web search tool - ONLY use as fallback when book search fails."""

from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search(query: str) -> str:
    """
    Search the web for external information.
    
    USE ONLY AS FALLBACK when:
    - Book search (semantic_search) returned no/insufficient results
    - Query is about external topics not in books (current events, general knowledge)
    
    DO NOT use for book-related topics.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)
        
        if not results:
            return "No web results found."
        
        formatted = "Web Search Results:\n"
        for i, r in enumerate(results, 1):
            formatted += f"\n{i}. {r.get('title', 'No title')}\n"
            formatted += f"   {r.get('body', 'No description')}\n"
            formatted += f"   Source: {r.get('href', '')}\n"
        
        return formatted
    except Exception as e:
        return f"Web search error: {str(e)}"