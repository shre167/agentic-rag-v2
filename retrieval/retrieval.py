import re

from .reranking import rerank_passages
from .vector_store import get_collection

MAX_DISTANCE = 0.65

NO_SOURCE_ANSWER = (
    "I couldn't find relevant information in the selected knowledge base."
)


def retrieve_chunks(
    kb_name: str,
    query: str,
    k: int = 5,
    max_distance: float = MAX_DISTANCE,
    book: str | None = None,
    chapter: str | None = None,
    section: str | None = None,
    page: int | None = None,
) -> list[dict]:
    """
    Retrieve passages from a specific knowledge base.
    Supports optional chapter, section and page filtering.
    """

    collection = get_collection(kb_name)

    if collection.count() == 0:
        return []

    stored = None
    where_conditions = []
    where = None

    book = book.strip() if book else None
    chapter = chapter.strip() if chapter else None
    section = section.strip() if section else None

    if book or chapter or section or page is not None:
        stored = collection.get(include=["metadatas"])

    if book and stored:
        canonical_book = next(
            (
                metadata.get("book", "")
                for metadata in stored["metadatas"]
                if metadata.get("book", "").casefold() == book.casefold()
            ),
            None,
        )
        if canonical_book is None:
            partial_matches = sorted(
                {
                    metadata.get("book", "")
                    for metadata in stored["metadatas"]
                    if book.casefold() in metadata.get("book", "").casefold()
                }
            )
            if len(partial_matches) != 1:
                return []
            canonical_book = partial_matches[0]
        where_conditions.append({"book": {"$eq": canonical_book}})

    # Chapter filter
    if chapter and stored:
        requested_number = re.search(
            r"\bchapter\s*(\d+)\b",
            chapter,
            re.IGNORECASE,
        )

        canonical_chapter = next(
            (
                metadata.get("chapter", "")
                for metadata in stored["metadatas"]
                if (
                    metadata.get("chapter", "").casefold()
                    == chapter.casefold()
                    or (
                        requested_number
                        and (
                            match := re.search(
                                r"\bchapter\s*(\d+)\b",
                                metadata.get("chapter", ""),
                                re.IGNORECASE,
                            )
                        )
                        and match.group(1)
                        == requested_number.group(1)
                    )
                )
            ),
            None,
        )

        if canonical_chapter is None:
            return []

        where_conditions.append(
            {"chapter": {"$eq": canonical_chapter}}
        )

    # Section filter
    if section and stored:
        canonical_section = next(
            (
                metadata.get("section", "")
                for metadata in stored["metadatas"]
                if metadata.get("section", "").casefold()
                == section.casefold()
            ),
            None,
        )

        if canonical_section is None:
            return []

        where_conditions.append(
            {"section": {"$eq": canonical_section}}
        )

    # Page filter
    if page is not None:
        where_conditions.extend(
            [
                {"page_start": {"$lte": page}},
                {"page_end": {"$gte": page}},
            ]
        )

    if where_conditions:
        where = (
            where_conditions[0]
            if len(where_conditions) == 1
            else {"$and": where_conditions}
        )

    scoped_count = (
        len(collection.get(where=where, include=[])["ids"])
        if where
        else collection.count()
    )

    if scoped_count == 0:
        return []

    candidate_count = min(
        scoped_count,
        max(k * 4, 20),
    )

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

    # Fallback if strict semantic filtering returns nothing
    if not relevant and where:
        fallback = collection.get(
            where=where,
            include=[
                "documents",
                "metadatas",
            ],
            limit=max(k * 4, 20),
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