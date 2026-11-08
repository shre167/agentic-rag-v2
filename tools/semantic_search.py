 
from langchain.tools import tool
 
from retrieval.retrieval import retrieve_chunks
from tools.results import tool_result
 
 
@tool
def semantic_search(
    query: str,
    kb_name: str,
    book: str = "",
    k: int = 5,
) -> str:
    """
    Semantic search across the entire selected knowledge base.
 
    Use when the user asks about a concept without specifying
    a particular chapter or section.
    """
 
    search_input = (
        f"{query} [kb: {kb_name}, book: {book}]"
        if book
        else f"{query} [kb: {kb_name}]"
    )
 
    passages = retrieve_chunks(
        query=query,
        kb_name=kb_name,
        k=k,
        book=book or None,
        chapter=None,
        section=None,
    )
 
    if not passages:
        return tool_result(
            "semantic_search",
            search_input,
            "No relevant passages found",
            [],
        )
 
    return tool_result(
        "semantic_search",
        search_input,
        f"Found {len(passages)} supporting passages",
        passages,
    )
 