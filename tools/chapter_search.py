from langchain.tools import tool
from retrieval.retrieval import indexed_collection, retrieve_chunks
from tools.results import tool_result


@tool
def chapter_search(
    chapter: str,
    query: str,
    kb_name: str,
    book: str = "",
    k: int = 5,
) -> str:
    """
    Search within a specific chapter.
    Use only when the chapter is known.
    """

    stored = indexed_collection(kb_name).get(include=["metadatas"])
    if book:
        stored["metadatas"] = [
            metadata
            for metadata in stored["metadatas"]
            if book.casefold() in metadata.get("book", "").casefold()
        ]
    requested_number = "".join(character for character in chapter if character.isdigit())
    chapter_exists = any(
        metadata.get("chapter", "").casefold() == chapter.strip().casefold()
        or (
            requested_number
            and "".join(
                character
                for character in metadata.get("chapter", "")
                if character.isdigit()
            )
            == requested_number
        )
        for metadata in stored["metadatas"]
    )
    if not chapter_exists:
        return tool_result(
            "chapter_search",
            f"{kb_name}: {book + ': ' if book else ''}{chapter}: {query}",
            "No matching chapter found",
        )

    passages = retrieve_chunks(
        query=query,
        kb_name=kb_name,
        k=k,
        book=book or None,
        chapter=chapter,
        section=None,
    )

    if not passages:
        return tool_result(
            "chapter_search",
            f"{kb_name}: {book + ': ' if book else ''}{chapter}: {query}",
            "No supporting passages found",
        )

    return tool_result(
        "chapter_search",
        f"{kb_name}: {book + ': ' if book else ''}{chapter}: {query}",
        f"Found {len(passages)} passages",
        passages,
    )
