"""
Retrieval Engine

Core retrieval logic: query ChromaDB with metadata filtering.

Pipeline:
  1. Parse query parameters (kb_name, page, chapter, section)
  2. Build ChromaDB where clause (filters)
  3. Query ChromaDB with semantic similarity + filters
  4. Rerank results by relevance
  5. Return top-k passages with metadata

Why separate from tools?
  - Keeps retrieval logic reusable
  - Tools call this, app.py can call this directly
  - Clean separation of concerns
"""

import re

from retrieval.vector_store import get_collection
from retrieval.reranking import rerank_passages


# Default response when nothing found
NO_SOURCE_ANSWER = "I couldn't find relevant information in the selected knowledge base."

# Maximum semantic distance (0-1 scale, lower = more relevant)
MAX_DISTANCE = 0.65


def retrieve_chunks(
    kb_name: str,
    query: str,
    k: int = 5,
    max_distance: float = MAX_DISTANCE,
    chapter: str = None,
    section: str = None,
    page: int = None,
) -> list:
    """
    Retrieve passages from knowledge base with optional metadata filtering.
    
    Pipeline:
      1. Get ChromaDB collection for kb_name
      2. Build where clause from metadata filters (page, chapter, section)
      3. Query with semantic similarity + where clause
      4. Rerank results by relevance
      5. Return top-k passages
    
    Args:
        kb_name (str): "kb1" or "kb2"
        query (str): Search text
        k (int): Number of results
        max_distance (float): Max semantic distance (higher = more lenient)
        chapter (str, optional): e.g., "chapter 3"
        section (str, optional): e.g., "deep work"
        page (int, optional): Page number
    
    Returns:
        list: Passages with metadata
        
    Why return early if no collection?
      - Invalid KB name → no collection → can't retrieve
      - Better to fail fast than query empty collection
    """
    
    # Get ChromaDB collection
    collection = get_collection(kb_name)
    
    if collection.count() == 0:
        return []
    
    # Build metadata filters (where clause)
    where_conditions = []
    
    # Page filter: match chunks where page_start <= target_page <= page_end
    if page is not None:
        where_conditions.extend([
            {"page_start": {"$lte": page}},  # Chunk must start before or at target
            {"page_end": {"$gte": page}},     # Chunk must end after or at target
        ])
    
    # Chapter filter: exact match on chapter metadata
    if chapter:
        chapter = chapter.strip()
        # Try to find canonical chapter name (handle "chapter 3" vs "Chapter 3")
        stored = collection.get(include=["metadatas"])
        requested_number = re.search(r"\bchapter\s*(\d+)\b", chapter, re.IGNORECASE)
        
        canonical_chapter = next(
            (
                m.get("chapter", "")
                for m in stored["metadatas"]
                if (
                    m.get("chapter", "").casefold() == chapter.casefold()
                    or (
                        requested_number
                        and (match := re.search(
                            r"\bchapter\s*(\d+)\b",
                            m.get("chapter", ""),
                            re.IGNORECASE,
                        ))
                        and match.group(1) == requested_number.group(1)
                    )
                )
            ),
            None,
        )
        
        if canonical_chapter is None:
            return []
        
        where_conditions.append({"chapter": {"$eq": canonical_chapter}})
    
    # Section filter: exact match on section metadata
    if section:
        section = section.strip()
        stored = collection.get(include=["metadatas"])
        
        canonical_section = next(
            (
                m.get("section", "")
                for m in stored["metadatas"]
                if m.get("section", "").casefold() == section.casefold()
            ),
            None,
        )
        
        if canonical_section is None:
            return []
        
        where_conditions.append({"section": {"$eq": canonical_section}})
    
    # Combine conditions with AND logic
    where = None
    if where_conditions:
        where = (
            where_conditions[0]
            if len(where_conditions) == 1
            else {"$and": where_conditions}
        )
    
    # Count matching chunks
    scoped_count = (
        len(collection.get(where=where, include=[])["ids"])
        if where
        else collection.count()
    )
    
    if scoped_count == 0:
        return []
    
    # Query with filters
    candidate_count = min(scoped_count, max(k * 4, 20))
    
    results = collection.query(
        query_texts=[query],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    
    # Build passages with metadata
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
    
    # Filter by distance threshold
    relevant = [p for p in passages if p["distance"] <= max_distance]
    
    # Fallback: if strict filtering returned nothing, use looser threshold
    if not relevant and where:
        fallback = collection.get(
            where=where,
            include=["documents", "metadatas"],
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
    
    # Rerank by relevance
    return rerank_passages(
        question=query,
        passages=relevant,
        limit=k,
    )
