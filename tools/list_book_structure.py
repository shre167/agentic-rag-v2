from langchain.tools import tool
from retrieval.vector_store import get_collection
from tools.results import tool_result


@tool
def list_book_structure(kb_name: str, book: str = "") -> str:
    """
    Return all chapters and sections in the indexed book.
    Useful when the agent needs to discover where a topic
    might be located before searching.
    """

    collection = get_collection(kb_name)
    if book:
        metadata = collection.get(include=["metadatas"])["metadatas"]
        matches = {
            item.get("book", "")
            for item in metadata
            if book.casefold() in item.get("book", "").casefold()
        }
        if len(matches) != 1:
            return tool_result(
                "list_book_structure",
                f"{kb_name}: {book}",
                "Book not found or title is ambiguous",
            )
        data = collection.get(
            where={"book": {"$eq": next(iter(matches))}},
            include=["metadatas"],
        )
    else:
        data = collection.get(include=["metadatas"])

    headings = sorted(
        {
            f"{m.get('book', 'Unknown')} -> {m.get('chapter')} -> {m.get('section')}"
            for m in data["metadatas"]
        }
    )

    return tool_result(
        "list_book_structure",
        f"{kb_name}: {book}" if book else kb_name,
        f"Found {len(headings)} headings",
        items=headings,
    )