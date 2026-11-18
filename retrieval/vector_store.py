import hashlib
from pathlib import Path
from typing import Dict, Any

import chromadb

from settings import DB_PATH

_CLIENT = None
_COLLECTIONS: Dict[str, Any] = {}


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=str(DB_PATH))
    return _CLIENT


def get_collection(kb_name: str):
    if kb_name not in _COLLECTIONS:
        client = _get_client()
        _COLLECTIONS[kb_name] = client.get_or_create_collection(
            name=kb_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTIONS[kb_name]


def index_chunks(
    chunks: list[tuple[str, dict]],
    source: Path,
    kb_name: str,
) -> int:
    """
    Index chunks into a specific knowledge base.
    """

    collection = get_collection(kb_name)

    source = source.resolve()

    source_id = hashlib.sha256(
        str(source).encode("utf-8")
    ).hexdigest()[:12]

    source_value = str(source)

    old_ids = collection.get(
        where={"source": source_value},
        include=[]
    )["ids"]

    ids = [
        f"{source_id}-{index:05d}"
        for index in range(len(chunks))
    ]

    if chunks:
        collection.upsert(
            ids=ids,
            documents=[
                text
                for text, _ in chunks
            ],
            metadatas=[
                {
                    **metadata,
                    "source": source_value,
                }
                for _, metadata in chunks
            ],
        )

    stale_ids = list(
        set(old_ids) - set(ids)
    )

    if stale_ids:
        collection.delete(ids=stale_ids)

    try:
        from retrieval.retrieval import clear_metadata_cache
        clear_metadata_cache()
    except Exception:
        pass

    try:
        from retrieval.kb_registry import clear_kb_registry_cache
        clear_kb_registry_cache()
    except Exception:
        pass

    try:
        from tools.list_books import clear_book_cache
        clear_book_cache()
    except Exception:
        pass

    return len(chunks)


def get_chunks(kb_name: str):
    return get_collection(kb_name).get(
        include=["documents", "metadatas"]
    )


def list_knowledge_bases():
    client = _get_client()
    collections = client.list_collections()

    return sorted(
        collection.name
        for collection in collections
    )


def delete_knowledge_base(kb_name: str) -> bool:
    """
    Completely delete a knowledge base:
    1. Removes collection and all vector chunks from ChromaDB.
    2. Drops in-memory collection cache.
    3. Removes its profile from disk (kb_profiles/book_profiles.json).
    4. Invalidates all retrieval, routing, and book discovery caches.
    """
    kb_name = kb_name.strip()
    client = _get_client()

    try:
        client.delete_collection(name=kb_name)
    except Exception as e:
        print(f"ChromaDB collection deletion note for '{kb_name}': {e}")

    # Remove from in-memory cache
    if kb_name in _COLLECTIONS:
        del _COLLECTIONS[kb_name]

    # Clean up profile from disk and registry
    try:
        from retrieval.kb_registry import delete_book_profile
        delete_book_profile(kb_name)
    except Exception:
        pass

    # Clear all caches
    try:
        from retrieval.retrieval import clear_metadata_cache
        clear_metadata_cache()
    except Exception:
        pass

    try:
        from retrieval.kb_registry import clear_kb_registry_cache
        clear_kb_registry_cache()
    except Exception:
        pass

    try:
        from tools.list_books import clear_book_cache
        clear_book_cache()
    except Exception:
        pass

    return True
