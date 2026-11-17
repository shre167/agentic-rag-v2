"""
Semantic Search Tool with Dynamic KB Discovery & Semantic Router
==================================================================
Combines dynamic knowledge base discovery, embedding-based candidate scoring,
and LLM routing into a single unified search tool.

Features:
- Dynamically discovers all ingested KBs in ChromaDB.
- Uses semantic router to route queries to the right KB (or multiple KBs for comparisons).
- Executes Dense Search + BM25 + RRF reranking via retrieve_chunks.
- Detects irrelevant queries and returns a NO_MATCH result so the agent can invoke web_search.
- Preserves direct kb_name parameter for backward compatibility.
"""

import re
import time
from langchain.tools import tool

from retrieval.retrieval import retrieve_chunks
from retrieval.semantic_router import route_query
from tools.results import tool_result


@tool
def semantic_search(
    query: str,
    kb_name: str = "",
    book: str = "",
    chapter: str = "",
    section: str = "",
    page: int | str | None = None,
    line: int | str | None = None,
    k: int = 5,
) -> str:
    """
    Metadata-aware semantic search across indexed book collections.
    Dynamically discovers all knowledge bases, uses an embedding + LLM router to select
    the appropriate collection(s), and retrieves relevant passages with citations.

    If the query cannot be answered by any indexed book, it returns a NO_MATCH signal
    so the agent can invoke web_search.

    Args:
        query: Search term or concept (e.g. "Cal Newport rules for deep work" or "page 10 line 15")
        kb_name: Optional specific KB name (e.g. 'kb1', 'kb2'). If omitted, automatically routed.
        book: Optional book title filter.
        chapter: Optional chapter name or number (e.g. "chapter 3" or "3").
        section: Optional section name.
        page: Optional page number (e.g. 50).
        line: Optional line number.
        k: Number of passages to retrieve per collection (default 5).

    Returns:
        Structured JSON string containing passages, metadata, citations, and routing reasoning.
    """
    start_time = time.perf_counter()

    parsed_page: int | None = None
    if page is not None:
        if isinstance(page, int):
            parsed_page = page
        elif isinstance(page, str):
            match = re.search(r"\d+", page)
            if match:
                parsed_page = int(match.group())

    parsed_line: int | None = None
    if line is not None:
        if isinstance(line, int):
            parsed_line = line
        elif isinstance(line, str):
            match = re.search(r"\d+", line)
            if match:
                parsed_line = int(match.group())

    # Auto-extract page from query if not explicitly passed
    if parsed_page is None:
        page_match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
        if page_match:
            parsed_page = int(page_match.group(1))

    # Auto-extract line from query if not explicitly passed
    if parsed_line is None:
        line_match = re.search(r"\bline\s*(\d+)\b", query, re.IGNORECASE)
        if line_match:
            parsed_line = int(line_match.group(1))

    # Auto-extract chapter from query if not explicitly passed
    if not chapter:
        WORD_NUMS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
        ch_match = re.search(r"\b(chapter|rule)\s+(?:#\s*)?(\d+|one|two|three|four|five)\b", query, re.IGNORECASE)
        if ch_match:
            kind, val = ch_match.group(1).title(), ch_match.group(2).lower()
            chapter = f"{kind} {WORD_NUMS.get(val, val)}"

    # ─────────────────────────────────────────────────────────────
    # CASE 1: Automatic Routing via Semantic Retrieval Router
    # ─────────────────────────────────────────────────────────────
    if not kb_name or kb_name.strip().lower() in {"auto", "all", "none", ""}:
        route = route_query(query)

        # Requirement 5: Handle irrelevant queries
        if not route.get("is_match") or not route.get("selected_kbs"):
            latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
            reason = route.get("reason", "Query is outside the scope of all indexed books.")
            return tool_result(
                "semantic_search",
                query,
                f"NO_MATCH: No relevant knowledge base found. {reason} Recommend using web_search.",
                passages=[],
                latency_ms=latency_ms,
            )

        selected_kbs = route["selected_kbs"]
        sub_queries = route.get("sub_queries", {})
        router_chapter = chapter or route.get("chapter")
        router_page = parsed_page if parsed_page is not None else route.get("page")
        router_line = parsed_line if parsed_line is not None else route.get("line")

        all_passages = []
        for target_kb in selected_kbs:
            kb_query = sub_queries.get(target_kb, query)
            passages = retrieve_chunks(
                query=kb_query,
                kb_name=target_kb,
                k=k,
                book=book or None,
                chapter=router_chapter or None,
                section=section or None,
                page=router_page,
                line=router_line,
            )
            all_passages.extend(passages)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        kbs_str = ", ".join(selected_kbs)
        search_desc = f'"{query}" [routed to: {kbs_str}]'

        if not all_passages:
            if router_page is not None:
                return tool_result(
                    "semantic_search",
                    search_desc,
                    f"EMPTY_PAGE: Page {router_page} has no indexed text in ({kbs_str}). In the PDF it is likely an image, illustration, cover art, or blank page.",
                    passages=[],
                    latency_ms=latency_ms,
                )
            return tool_result(
                "semantic_search",
                search_desc,
                f"NO_MATCH: Collections ({kbs_str}) had no passages matching '{query}'. Try another KB or invoke web_search.",
                passages=[],
                latency_ms=latency_ms,
            )

        # ─────────────────────────────────────────────────────────────
        # Evidence Relevance & Confidence Evaluation
        # ─────────────────────────────────────────────────────────────
        max_rerank = max((p.get("rerank_score", 0.0) for p in all_passages), default=0.0)
        avg_dist = sum(float(p.get("distance", 1.0)) for p in all_passages) / len(all_passages) if all_passages else 1.0

        if max_rerank >= 0.40 and avg_dist <= 0.48:
            confidence = "high"
        elif max_rerank >= 0.22 or avg_dist <= 0.58:
            confidence = "medium"
        else:
            confidence = "low"

        if confidence == "low":
            summary_msg = (
                f"INSUFFICIENT_EVIDENCE: Retrieved {len(all_passages)} passages from [{kbs_str}] but relevance confidence is LOW "
                f"(max_score={max_rerank:.2f}, avg_distance={avg_dist:.2f}). "
                f"If insufficient to answer, retry another knowledge base or invoke web_search."
            )
        else:
            summary_msg = (
                f"Retrieved {len(all_passages)} passages across [{kbs_str}] with {confidence.upper()} confidence "
                f"(max_score={max_rerank:.2f}): {route.get('reason', '')}"
            )

        return tool_result(
            "semantic_search",
            search_desc,
            summary_msg,
            passages=all_passages,
            latency_ms=latency_ms,
        )

    # ─────────────────────────────────────────────────────────────
    # CASE 2: Explicit kb_name Provided (Direct Retrieval / Backward Compatible)
    # ─────────────────────────────────────────────────────────────
    search_desc = f'"{query}" [kb: {kb_name}'
    if parsed_page is not None:
        search_desc += f", page: {parsed_page}"
    if parsed_line is not None:
        search_desc += f", line: {parsed_line}"
    if chapter:
        search_desc += f", chapter: {chapter}"
    if section:
        search_desc += f", section: {section}"
    search_desc += "]"

    passages = retrieve_chunks(
        query=query,
        kb_name=kb_name,
        k=k,
        book=book or None,
        chapter=chapter or None,
        section=section or None,
        page=parsed_page,
        line=parsed_line,
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    if not passages:
        if parsed_page is not None:
            return tool_result(
                "semantic_search",
                search_desc,
                f"EMPTY_PAGE: Page {parsed_page} in {kb_name} contains no indexed text (it is likely an image, illustration, cover art, or blank page in the PDF).",
                [],
                latency_ms=latency_ms,
            )
        return tool_result(
            "semantic_search",
            search_desc,
            f"No relevant passages found in {kb_name}.",
            [],
            latency_ms=latency_ms,
        )

    return tool_result(
        "semantic_search",
        search_desc,
        f"Found {len(passages)} supporting passages",
        passages,
        latency_ms=latency_ms,
    )
