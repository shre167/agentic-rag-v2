import hashlib
from pathlib import Path

import chromadb

from settings import COLLECTION_NAME, DB_PATH


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[tuple[str, dict]], source: Path) -> int:
    """Upsert this source's chunks and remove stale chunks from earlier runs."""
    collection = get_collection()
    source = source.resolve()
    source_id = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    source_value = str(source)
    old_ids = collection.get(where={"source": source_value}, include=[])["ids"]
    ids = [f"{source_id}-{index:05d}" for index in range(len(chunks))]

    if chunks:
        collection.upsert(
            ids=ids,
            documents=[text for text, _ in chunks],
            metadatas=[{**metadata, "source": source_value} for _, metadata in chunks],
        )

    stale_ids = list(set(old_ids) - set(ids))
    if stale_ids:
        collection.delete(ids=stale_ids)
    return len(chunks)


def get_chunks():
    """Return all stored chunk IDs, text, and metadata for inspection."""
    return get_collection().get(include=["documents", "metadatas"])
