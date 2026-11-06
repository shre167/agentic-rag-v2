from langchain.tools import tool

from retrieval.retrieval import retrieve_chunks
from tools.results import tool_result


@tool
def page_search(
    page: int, kb_name: str, query: str = "", book: str = "", k: int = 5
) -> str:
    """Search text on a physical PDF page number, optionally using a topic query."""
    if page < 1:
        return tool_result("page_search", str(page), "Page number must be positive")

    search_query = query.strip() or "page content"
    passages = retrieve_chunks(
        query=search_query,
        kb_name=kb_name,
        k=k,
        book=book or None,
        page=page,
    )
    if not passages:
        return tool_result(
            "page_search",
            f"kb={kb_name}, page={page}, book={book}, query={query}",
            "No passages on that page",
        )

    return tool_result(
        "page_search",
        f"kb={kb_name}, page={page}, book={book}, query={query}",
        f"Found {len(passages)} passages on PDF page {page}",
        passages,
    )
