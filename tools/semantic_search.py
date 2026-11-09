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
    Semantic search with metadata-aware filtering.
    
    Automatically handles page, chapter, and section filtering based on query.
    
    Args:
        query: Search term or concept
        kb_name: Knowledge base (kb1 or kb2)
        book: Book name (optional)
        chapter: Chapter name/number (optional, e.g., "chapter 3")
        section: Section name (optional, e.g., "deep work")
        page: Page number (optional, for page-specific search)
        k: Number of results (default 5)
    
    Returns:
        Passages with metadata: page_start, page_end, chapter, section
    """
    
    search_desc = f'"{query}" [kb: {kb_name}'
    if page:
        search_desc += f", page: {page}"
    if chapter:
        search_desc += f", chapter: {chapter}"
    if section:
        search_desc += f", section: {section}"
    search_desc += "]"
    
    passages = retrieve_chunks(
        query=query,
        kb_name=kb_name,
        k=k,
        book=book or None,
        chapter=chapter or None,
        section=section or None,
        page=page,
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