from langchain.tools import tool
from retrieval.retrieval import retrieve_chunks
from tools.results import tool_result


@tool
def section_search(
    section: str,
    query: str,
    kb_name: str,
    book: str = "",
    k: int = 5,
) -> str:
    """
    Search within a specific section.
    Use only when the section name is known.
    """

    passages = retrieve_chunks(
        query=query,
        kb_name=kb_name,
        k=k,
        book=book or None,
        chapter=None,
        section=section,
    )

    if not passages:
        return tool_result(
            "section_search",
            f"{kb_name}: {book + ': ' if book else ''}{section}: {query}",
            "No relevant passages",
        )

    return tool_result(
        "section_search",
        f"{kb_name}: {book + ': ' if book else ''}{section}: {query}",
        f"Found {len(passages)} passages",
        passages,
    )