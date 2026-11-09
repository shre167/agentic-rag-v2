import re
from typing import Dict, Any, List

from .reranking import rerank_passages
from .vector_store import get_collection

MAX_DISTANCE = 0.65

NO_SOURCE_ANSWER = (
    "I couldn't find relevant information in the selected knowledge base."
)

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
    Retrieve passages from a specific knowledge base.
    Optimized for sub-second latency with cached metadata indexing and fast-path lookups.
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

    # Only inspect metadata index if book, chapter, or section needs resolving
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

    # Page filter
    if page is not None:
        where_conditions.extend(
            [
                {"page_start": {"$lte": page}},
                {"page_end": {"$gte": page}},
            ]
        )

    # Line filter
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

    # Fast path: direct retrieval when query is a lookup for page/line/structure
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

    # Semantic search with where filter
    candidate_count = min(total_count, max(k * 4, 15))

    results = collection.query(
        query_texts=[query],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
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

    relevant = [
        passage
        for passage in passages
        if passage["distance"] <= max_distance
    ]

    # Fallback if strict semantic distance returned nothing
    if not relevant and where:
        fallback = collection.get(
            where=where,
            include=[
                "documents",
                "metadatas",
            ],
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


def indexed_collection(
    kb_name: str,
):
    """
    Return a KB collection for inspection/debugging.
    """
    return get_collection(kb_name)
