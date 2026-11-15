import json
import time

from langchain.tools import tool

from retrieval.vector_store import get_collection, list_knowledge_bases

_BOOK_CACHE: list[tuple[str, str]] | None = None


def clear_book_cache():
    """Clear the in-memory cache of indexed books to force discovery on next call."""
    global _BOOK_CACHE
    _BOOK_CACHE = None


@tool
def list_books() -> str:
    """
    Discover all available knowledge bases and the books indexed in each one.

    When searching books, call this before semantic_search to discover which
    kb_name to pass for each book. Do NOT call this for greetings or general questions.

    Returns a JSON object with:
      - routing_map: dict mapping kb_name -> book title
      - instruction: explicit guidance on how to use kb_name in semantic_search
      - items: list of "kb_name: Book Title" strings
    """
    start_time = time.perf_counter()
    global _BOOK_CACHE
    if _BOOK_CACHE is None:
        books = []
        for kb_name in list_knowledge_bases():
            col = get_collection(kb_name)
            col_book = (getattr(col, "metadata", None) or {}).get("book")
            if col_book:
                books.append((kb_name, col_book))
                continue

            meta_sample = col.get(limit=5, include=["metadatas"])["metadatas"]
            titles = sorted({item.get("book", "Unknown") for item in meta_sample if item.get("book")})
            if not titles:
                # Fallback to full check if sample had no book
                metadatas = col.get(include=["metadatas"])["metadatas"]
                titles = sorted({item.get("book", "Unknown") for item in metadatas if item.get("book")})
            for title in titles:
                books.append((kb_name, title))
        _BOOK_CACHE = books
    else:
        books = _BOOK_CACHE

    routing_map = {kb: title for kb, title in books}
    items = [f"{kb}: {title}" for kb, title in books]

    routing_instructions = (
        "ROUTING INSTRUCTIONS: "
        "To search a book, call semantic_search with the kb_name shown above. "
        "For multi-book or comparative questions, call semantic_search ONCE PER kb_name. "
        f"Available kb_names: {list(routing_map.keys())}. "
        "Do NOT search only one kb if the question spans multiple books."
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return json.dumps(
        {
            "tool": "list_books",
            "input": "",
            "result": f"Found {len(books)} books across knowledge bases",
            "routing_map": routing_map,
            "instruction": routing_instructions,
            "passages": [],
            "items": items,
            "latency_ms": latency_ms,
        },
        ensure_ascii=False,
    )
