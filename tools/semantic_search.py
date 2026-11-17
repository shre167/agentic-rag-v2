"""
Simple Semantic Search Tool with Auto-Routing
==============================================
- Automatically discovers available KBs from ChromaDB
- Searches all KBs if no book specified
- Searches specific KB if book name mentioned
- Works with any number of books (dynamic)
"""

import re
from langchain.tools import tool

from retrieval.vector_store import list_knowledge_bases
from retrieval.retrieval import retrieve_chunks
from tools.results import tool_result


# KB Book name mappings - automatically discovered
KB_BOOK_MAP = {
    "kb1": "mind management",
    "kb2": "deep work",
}


def _get_available_kbs():
    """Get list of all available KBs from ChromaDB."""
    return list_knowledge_bases()


def _detect_kbs_from_query(query: str) -> list:
    """
    Simple logic to detect which KB(s) to search.
    - If book name mentioned → search that KB
    - If comparison words → search all KBs
    - Otherwise → search all KBs
    """
    query_lower = query.lower()
    selected_kbs = []
    
    # Check if any book is mentioned
    for kb_name, book_keywords in KB_BOOK_MAP.items():
        if book_keywords in query_lower:
            selected_kbs.append(kb_name)
    
    # Check for comparison words
    comparison_words = ["compare", "both", "difference", "versus", "vs", "and"]
    is_comparison = any(word in query_lower for word in comparison_words)
    
    # If no specific book mentioned OR it's a comparison → search all KBs
    if not selected_kbs or is_comparison:
        available = _get_available_kbs()
        if available:
            selected_kbs = available
    
    return selected_kbs


@tool
def semantic_search(
    query: str,
    kb_name: str = "",
    k: int = 5,
) -> str:
    """
    Search books with automatic KB routing.
    
    Args:
        query: Search term or concept
        kb_name: Specific KB to search (optional). If empty, auto-detects.
        k: Number of results per KB (default 5)
    
    How routing works:
    - If kb_name provided → search that KB
    - If no kb_name → auto-detect which KB(s) to search
    - If book name in query → search that book
    - If comparison ("both", "compare") → search all KBs
    - Otherwise → search all KBs
    
    Returns:
        Passages with metadata from searched KB(s)
    """
    
    # Auto-extract page number from query
    parsed_page = None
    page_match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
    if page_match:
        parsed_page = int(page_match.group(1))
    
    # Determine which KBs to search
    if kb_name and kb_name.strip():
        # Explicit KB provided
        target_kbs = [kb_name.strip()]
    else:
        # Auto-detect from query
        target_kbs = _detect_kbs_from_query(query)
    
    if not target_kbs:
        return tool_result(
            "semantic_search",
            query,
            "No knowledge bases found.",
            [],
        )
    
    # Search each KB and collect results
    all_passages = []
    searched_kbs = []
    
    for kb in target_kbs:
        passages = retrieve_chunks(
            query=query,
            kb_name=kb,
            k=k,
            page=parsed_page,
        )
        
        if passages:
            all_passages.extend(passages)
            searched_kbs.append(kb)
    
    # Build search description
    kbs_str = ", ".join(searched_kbs) if searched_kbs else "none"
    search_desc = f'"{query}" [kb: {kbs_str}]'
    if parsed_page:
        search_desc += f", page: {parsed_page}"
    
    if not all_passages:
        return tool_result(
            "semantic_search",
            search_desc,
            f"No relevant passages found in {kbs_str}.",
            [],
        )
    
    return tool_result(
        "semantic_search",
        search_desc,
        f"Found {len(all_passages)} passages from {kbs_str}",
        all_passages,
    )