"""
Retrieval Engine

Core retrieval logic: query ChromaDB with metadata filtering (book, page, line, chapter, section).
Optimized for low-latency retrieval.
"""

import re
from typing import Dict, Any

from retrieval.vector_store import get_collection
from retrieval.reranking import rerank_passages


NO_SOURCE_ANSWER = "I couldn't find relevant information in the selected knowledge base."
MAX_DISTANCE = 0.65

_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_metadata_index(collection, kb_name: str) -> Dict[str, Any]:
    if kb_name not in _METADATA_CACHE:
        stored = collection.get(include=["metadatas"])
        metas = stored.get("metadatas", [])
        books = sorted({m.get("book", "") for m in metas if m.get("book")})
        chapters = sorted({m.get("chapter", "") for m in metas if m.get("chapter")})
        sections = sorted({m.get("section", "") for m in metas if m.get("section")})
        chapter_numbers = {
            str(m.get("chapter_number")): m.get("chapter", "")
            for m in metas
            if m.get("chapter_number") is not None and m.get("chapter")
        }
        _METADATA_CACHE[kb_name] = {
            "books": books,
            "chapters": chapters,
            "sections": sections,
            "chapter_numbers": chapter_numbers,
        }
    return _METADATA_CACHE[kb_name]


def retrieve_chunks(
    kb_name: str,
    query: str,
    k: int = 5,
    max_distance: float = MAX_DISTANCE,
    book: str | None = None,
    chapter: str | None = None,
    section: str | None = None,
    page: int | None = None,
    line: int | None = None,
) -> list[dict]:
    """
    Retrieve passages from knowledge base with optional metadata filtering.

    Args:
        kb_name (str): "kb1" or "kb2"
        query (str): Search text
        k (int): Number of results
        max_distance (float): Max semantic distance (higher = more lenient)
        book (str, optional): Book title or partial title
        chapter (str, optional): e.g., "chapter 3" or "3"
        section (str, optional): e.g., "deep work"
        page (int, optional): Page number
        line (int, optional): Line number

    Returns:
        list[dict]: Passages with metadata
    """

    collection = get_collection(kb_name)

    total_count = collection.count()
    if total_count == 0:
        return []

    where_conditions = []
    where = None

    book = book.strip() if book else None
    chapter = chapter.strip() if chapter else None
    section = section.strip() if section else None

    if book or chapter or section:
        meta_index = _get_metadata_index(collection, kb_name)

        if book:
            canonical_book = next(
                (b for b in meta_index["books"] if b.casefold() == book.casefold()),
                None,
            )
            if canonical_book is None:
                partial_matches = sorted(
                    {b for b in meta_index["books"] if book.casefold() in b.casefold()}
                )
                if len(partial_matches) == 1:
                    canonical_book = partial_matches[0]
            if canonical_book is None:
                return []
            where_conditions.append({"book": {"$eq": canonical_book}})

        if chapter:
            requested_number = re.search(
                r"\b(?:chapter\s*)?(\d+)\b",
                chapter,
                re.IGNORECASE,
            )
            canonical_chapter = next(
                (
                    c
                    for c in meta_index["chapters"]
                    if (
                        c.casefold() == chapter.casefold()
                        or (
                            requested_number
                            and (
                                (
                                    match := re.search(
                                        r"\b(?:chapter\s*)?(\d+)\b",
                                        c,
                                        re.IGNORECASE,
                                    )
                                )
                                and match.group(1) == requested_number.group(1)
                            )
                        )
                    )
                ),
                None,
            )
            if canonical_chapter is None and requested_number:
                canonical_chapter = meta_index["chapter_numbers"].get(
                    requested_number.group(1)
                )

            if canonical_chapter is None:
                return []

            where_conditions.append({"chapter": {"$eq": canonical_chapter}})

        if section:
            canonical_section = next(
                (
                    s
                    for s in meta_index["sections"]
                    if s.casefold() == section.casefold()
                ),
                None,
            )
            if canonical_section is None:
                partial_sections = sorted(
                    {
                        s
                        for s in meta_index["sections"]
                        if section.casefold() in s.casefold()
                    }
                )
                if len(partial_sections) == 1:
                    canonical_section = partial_sections[0]

            if canonical_section is None:
                return []

            where_conditions.append({"section": {"$eq": canonical_section}})

    if page is not None:
        where_conditions.extend(
            [
                {"page_start": {"$lte": page}},
                {"page_end": {"$gte": page}},
            ]
        )

    if line is not None:
        where_conditions.extend(
            [
                {"line_start": {"$lte": line}},
                {"line_end": {"$gte": line}},
            ]
        )

    if where_conditions:
        where = (
            where_conditions[0]
            if len(where_conditions) == 1
            else {"$and": where_conditions}
        )

    is_direct_lookup = bool(
        where
        and (
            not query.strip()
            or re.search(
                r"^\s*(what('?s| is)? (on|in|at) )?(page|line|chapter|section)\s*\d+.*\??\s*$",
                query.strip(),
                re.IGNORECASE,
            )
        )
    )

    if is_direct_lookup:
        lookup_res = collection.get(
            where=where,
            include=["documents", "metadatas"],
            limit=k * 2,
        )
        if lookup_res["ids"]:
            passages = [
                {
                    "id": chunk_id,
                    "text": text,
                    **metadata,
                    "distance": 0.0,
                }
                for chunk_id, text, metadata in zip(
                    lookup_res["ids"],
                    lookup_res["documents"],
                    lookup_res["metadatas"],
                )
            ]
            return passages[:k]

    candidate_count = min(total_count, max(k * 4, 15))

    results = collection.query(
        query_texts=[query],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    passages = [
        {
            "id": chunk_id,
            "text": text,
            **metadata,
            "distance": distance,
        }
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    relevant = [p for p in passages if p["distance"] <= max_distance]

    if not relevant and where:
        fallback = collection.get(
            where=where,
            include=["documents", "metadatas"],
            limit=max(k * 4, 15),
        )

        relevant = [
            {
                "id": chunk_id,
                "text": text,
                **metadata,
                "distance": 1.0,
            }
            for chunk_id, text, metadata in zip(
                fallback["ids"],
                fallback["documents"],
                fallback["metadatas"],
            )
        ]

    return rerank_passages(
        question=query,
        passages=relevant,
        limit=k,
    )
