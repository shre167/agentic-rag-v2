"""
Dynamic Knowledge Base Registry
================================
Provides dynamic discovery and profiling of all currently ingested knowledge bases.
Computes semantic embeddings for KB profiles to enable embedding-based candidate routing.
Whenever new books are ingested, the registry automatically discovers them.
"""

from typing import Dict, Any, List
import numpy as np

from retrieval.vector_store import get_collection, list_knowledge_bases

_REGISTRY_CACHE: Dict[str, Dict[str, Any]] | None = None
_KNOWN_COLLECTIONS: tuple[str, ...] | None = None


def clear_kb_registry_cache():
    """Clear cached KB profiles to force re-discovery on next query."""
    global _REGISTRY_CACHE, _KNOWN_COLLECTIONS
    _REGISTRY_CACHE = None
    _KNOWN_COLLECTIONS = None


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def get_kb_registry() -> Dict[str, Dict[str, Any]]:
    """
    Dynamically discover all ingested knowledge bases from ChromaDB and return their profiles.
    Automatically refreshes if the set of collections in ChromaDB changes.
    """
    global _REGISTRY_CACHE, _KNOWN_COLLECTIONS

    current_kbs = tuple(list_knowledge_bases())

    if _REGISTRY_CACHE is not None and _KNOWN_COLLECTIONS == current_kbs:
        return _REGISTRY_CACHE

    registry: Dict[str, Dict[str, Any]] = {}
    if not current_kbs:
        _REGISTRY_CACHE = {}
        _KNOWN_COLLECTIONS = current_kbs
        return _REGISTRY_CACHE

    # Use first available collection's embedding function (Chroma DefaultEmbeddingFunction)
    first_col = get_collection(current_kbs[0])
    embed_fn = getattr(first_col, "_embedding_function", None)

    profile_texts_to_embed = []
    kb_order = []

    for kb_name in current_kbs:
        try:
            col = get_collection(kb_name)
            count = col.count()
            if count == 0:
                continue

            # Extract book title from collection metadata or chunk sample
            col_meta = getattr(col, "metadata", None) or {}
            book_title = col_meta.get("book")

            # Gather all distinct chapters and sections using metadata index
            from retrieval.retrieval import _get_metadata_index
            meta_index = _get_metadata_index(col, kb_name)

            if not book_title and meta_index.get("books"):
                book_title = meta_index["books"][0]
            if not book_title:
                book_title = kb_name

            raw_chapters = [c for c in meta_index.get("chapters", []) if c != "Front matter"]
            raw_sections = [s for s in meta_index.get("sections", []) if s != "Front matter"]

            # Filter out boilerplate sections
            sections = [
                s for s in raw_sections
                if not any(b in s.upper() for b in ["ACKNOWLEDGEMENTS", "ABOUT THE AUTHOR", "BONUS MATERIAL", "ALSO BY"])
            ]

            profile_text = (
                f"Knowledge Base: {kb_name}. Book: {book_title}. "
                f"Chapters: {', '.join(raw_chapters[:12])}. "
                f"Topics and Sections: {', '.join(sections[:20])}."
            )

            registry[kb_name] = {
                "kb_name": kb_name,
                "book_title": book_title,
                "total_chunks": count,
                "chapters": raw_chapters,
                "sections": sections,
                "profile_text": profile_text,
                "embedding": None,
            }
            profile_texts_to_embed.append(profile_text)
            kb_order.append(kb_name)

        except Exception:
            continue

    # Batch compute embeddings for all KB profile texts
    if embed_fn and profile_texts_to_embed:
        try:
            embeddings = embed_fn(profile_texts_to_embed)
            for kb_name, emb in zip(kb_order, embeddings):
                registry[kb_name]["embedding"] = np.array(emb, dtype=np.float32)
        except Exception:
            pass

    _REGISTRY_CACHE = registry
    _KNOWN_COLLECTIONS = current_kbs
    return _REGISTRY_CACHE


def find_kb_candidates(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Use embeddings to score and rank available knowledge bases against the query.
    Returns candidates sorted by cosine similarity in descending order.
    """
    registry = get_kb_registry()
    if not registry:
        return []

    # Get embedding function from first collection
    first_kb = next(iter(registry.keys()))
    col = get_collection(first_kb)
    embed_fn = getattr(col, "_embedding_function", None)

    q_emb = None
    if embed_fn:
        try:
            q_emb = np.array(embed_fn([query_text])[0], dtype=np.float32)
        except Exception:
            q_emb = None

    candidates = []
    for kb_name, data in registry.items():
        similarity = 0.0
        if q_emb is not None and data.get("embedding") is not None:
            similarity = _cosine_similarity(q_emb, data["embedding"])
        else:
            # Fallback simple keyword heuristic if embedding unavailable
            q_lower = query_text.lower()
            title_lower = data["book_title"].lower()
            if title_lower in q_lower or any(word in q_lower for word in title_lower.split() if len(word) > 3):
                similarity = 0.6
            else:
                similarity = 0.1

        candidates.append({
            "kb_name": kb_name,
            "book_title": data["book_title"],
            "similarity": round(similarity, 4),
            "chapters": data["chapters"][:6],
            "sections": data["sections"][:6],
            "total_chunks": data["total_chunks"],
        })

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates[:top_k]
