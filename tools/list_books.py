from langchain.tools import tool

from retrieval.vector_store import get_collection, list_knowledge_bases
from tools.results import tool_result


@tool
def list_books() -> str:
    """List indexed book titles and their knowledge-base IDs."""
    books = []
    for kb_name in list_knowledge_bases():
        metadata = get_collection(kb_name).get(include=["metadatas"])["metadatas"]
        books.extend(
            (kb_name, title)
            for title in sorted({item.get("book", "Unknown") for item in metadata})
        )
    return tool_result(
        "list_books",
        "",
        f"Found {len(books)} books",
        items=[f"{kb_name}: {title}" for kb_name, title in books],
    )
