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
    page: int | str | None = None,
    line: int | str | None = None,
    k: int = 5,
) -> str:
    """
    Semantic search with metadata-aware filtering (page, line, chapter, section).

    Automatically handles page, line, chapter, and section filtering based on parameters or query.

    Args:
        query: Search term or concept
        kb_name: Knowledge base (kb1 or kb2)
        book: Book name (optional)
        chapter: Chapter name/number (optional, e.g., "chapter 3" or "3")
        section: Section name (optional, e.g., "deep work")
        page: Page number (optional, e.g. 50 or "50")
        line: Line number on page (optional, e.g. 10 or "10")
        k: Number of results (default 5)

    Returns:
        Passages with metadata: page_start, page_end, line_start, line_end, chapter, section
    """
    parsed_page: int | None = None
    if page is not None:
        if isinstance(page, int):
            parsed_page = page
        elif isinstance(page, str):
            match = re.search(r"\d+", page)
            if match:
                parsed_page = int(match.group())

    parsed_line: int | None = None
    if line is not None:
        if isinstance(line, int):
            parsed_line = line
        elif isinstance(line, str):
            match = re.search(r"\d+", line)
            if match:
                parsed_line = int(match.group())

    # Auto-extract page from query if not explicitly passed
    if parsed_page is None:
        page_match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
        if page_match:
            parsed_page = int(page_match.group(1))

    # Auto-extract line from query if not explicitly passed
    if parsed_line is None:
        line_match = re.search(r"\bline\s*(\d+)\b", query, re.IGNORECASE)
        if line_match:
            parsed_line = int(line_match.group(1))

    # Auto-extract chapter from query if not explicitly passed
    if not chapter:
        chap_match = re.search(r"\bchapter\s*(\d+)\b", query, re.IGNORECASE)
        if chap_match:
            chapter = f"Chapter {chap_match.group(1)}"

    search_desc = f'"{query}" [kb: {kb_name}'
    if parsed_page is not None:
        search_desc += f", page: {parsed_page}"
    if parsed_line is not None:
        search_desc += f", line: {parsed_line}"
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
        page=parsed_page,
        line=parsed_line,
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
