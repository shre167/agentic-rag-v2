import json

from langchain.tools import tool

from retrieval.vector_store import get_collection, list_knowledge_bases

_BOOK_CACHE: list[tuple[str, str]] | None = None


@tool
def list_books() -> str:
    """
    Discover all available knowledge bases and the books indexed in each one.

    Call this FIRST before any semantic_search call. The response tells you
    which kb_name to pass to semantic_search for each book.

    Returns a JSON object with:
      - routing_map: dict mapping kb_name -> book title
      - instruction: explicit guidance on how to use kb_name in semantic_search
      - items: list of "kb_name: Book Title" strings
    """
    global _BOOK_CACHE
    if _BOOK_CACHE is None:
        books = []
        for kb_name in list_knowledge_bases():
            col = get_collection(kb_name)
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

    return json.dumps(
        {
            "tool": "list_books",
            "input": "",
            "result": f"Found {len(books)} books across knowledge bases",
            "routing_map": routing_map,
            "instruction": routing_instructions,
            "passages": [],
            "items": items,
        },
        ensure_ascii=False,
    )
