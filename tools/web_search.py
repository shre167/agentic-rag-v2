"""Web search tool for external information."""

from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search(query: str) -> str:
    """
    Search the web for external information.
    
    Use ONLY when:
    - Query is about external info (not in books)
    - User asks for web information
    - Book search returns insufficient results
    
    Args:
        query: Search query
        
    Returns:
        Web search results
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