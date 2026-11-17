"""
Semantic Search Tool - Direct retrieval without complex routing
================================================================
Simple, reliable semantic search that queries ChromaDB directly.
"""

import re
from langchain.tools import tool

from retrieval.retrieval import retrieve_chunks
from tools.results import tool_result


@tool
def semantic_search(
    query: str,
    kb_name: str,
    book: str = "",
    chapter: str = "",
    section: str = "",
    page: int = None,
    k: int = 5,
) -> str:
    """
    Semantic search across book collections.
    
    Args:
        query: Search term or concept
        kb_name: Knowledge base (kb1 or kb2) - REQUIRED
        book: Book name filter (optional)
        chapter: Chapter filter (optional)
        section: Section filter (optional)
        page: Page number filter (optional)
        k: Number of results (default 5)
    
    Returns:
        Passages with metadata from the specified KB
    """
    
    # Auto-extract page from query if not provided
    parsed_page = page
    if parsed_page is None:
        page_match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
        if page_match:
            parsed_page = int(page_match.group(1))
    
    # Build search description
    search_desc = f'"{query}" [kb: {kb_name}'
    if parsed_page:
        search_desc += f", page: {parsed_page}"
    if chapter:
        search_desc += f", chapter: {chapter}"
    if section:
        search_desc += f", section: {section}"
    search_desc += "]"
    
    # Direct retrieval - no router
    passages = retrieve_chunks(
        query=query,
        kb_name=kb_name,
        k=k,
        book=book or None,
        chapter=chapter or None,
        section=section or None,
        page=parsed_page,
    )
    
    if not passages:
        return tool_result(
            "semantic_search",
            search_desc,
            "No relevant passages found",
            [],
        )
    
    return tool_result(
        "semantic_search",
        search_desc,
        f"Found {len(passages)} supporting passages",
        passages,
    )