"""
Semantic Search Tool with LLM-Based Profile Routing
=====================================================
- Routes queries using LLM reasoning over book profiles
- Searches ONLY the selected KB (no all-KB fallback)
- Returns NO_MATCH if no book is relevant
- Each book is stored in its own separate collection
"""

import json
import time
from langchain.tools import tool

from retrieval.semantic_router import route_query
from retrieval.retrieval import retrieve_chunks
from retrieval.kb_registry import get_kb_registry


@tool
def semantic_search(
    query: str,
    kb_name: str = "",
    k: int = 5,
) -> str:
    """
    Search books with LLM-based profile routing.

    Args:
        query: Search term or concept
        kb_name: Specific KB to search (optional). If empty, the LLM router
                 selects the best KB based on book profiles.
        k: Number of results (default 5)

    How routing works:
    - If kb_name provided → search that specific KB
    - If no kb_name → LLM analyzes book profiles to select the most relevant KB
    - If no book matches the query → returns NO_MATCH
    - Never searches all KBs as a fallback

    Returns:
        Passages with metadata from the selected KB, or NO_MATCH
    """
    start_time = time.time()

    # Determine which KB to search
    if kb_name and kb_name.strip():
        # Explicit KB provided — validate it exists
        registry = get_kb_registry()
        target_kb = kb_name.strip()
        if target_kb not in registry:
            result_data = {
                "tool": "semantic_search",
                "input": query,
                "result": f"NO_MATCH: Knowledge base '{target_kb}' not found.",
                "passages": [],
                "items": [],
                "tool_reasoning": f"Specified KB '{target_kb}' does not exist.",
                "latency_ms": (time.time() - start_time) * 1000,
            }
            return json.dumps(result_data, ensure_ascii=False)

        routing_reason = f"Explicit KB specified: {target_kb}"
        sub_query = query
        parsed_page = None
        parsed_line = None
        parsed_chapter = None

        import re
        page_match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
        if page_match:
            parsed_page = int(page_match.group(1))

    else:
        # LLM-based routing using book profiles
        route_result = route_query(query)

        if not route_result["is_match"]:
            result_data = {
                "tool": "semantic_search",
                "input": query,
                "result": f"NO_MATCH: {route_result['reason']}",
                "passages": [],
                "items": [],
                "tool_reasoning": route_result["reason"],
                "latency_ms": (time.time() - start_time) * 1000,
            }
            return json.dumps(result_data, ensure_ascii=False)

        target_kb = route_result["selected_kb"]
        sub_query = route_result["sub_query"]
        parsed_page = route_result.get("page")
        parsed_line = route_result.get("line")
        parsed_chapter = route_result.get("chapter")
        routing_reason = route_result["reason"]

    # Retrieve passages from the SINGLE selected KB only
    passages = retrieve_chunks(
        query=sub_query,
        kb_name=target_kb,
        k=k,
        page=parsed_page,
        line=parsed_line,
        chapter=parsed_chapter,
    )

    # Tag passages with KB metadata
    if passages:
        for p in passages:
            p["kb_name"] = target_kb

    search_desc = f'"{query}" [kb: {target_kb}]'
    if parsed_page:
        search_desc += f", page: {parsed_page}"

    if not passages:
        result_data = {
            "tool": "semantic_search",
            "input": search_desc,
            "result": f"No relevant passages found in {target_kb}.",
            "passages": [],
            "items": [],
            "tool_reasoning": routing_reason,
            "latency_ms": (time.time() - start_time) * 1000,
        }
        return json.dumps(result_data, ensure_ascii=False)

    # Format items for display
    items = [f"[{p['id']}] {p.get('book', '?')} - {p.get('section', '')}" for p in passages]

    result_data = {
        "tool": "semantic_search",
        "input": search_desc,
        "result": f"Found {len(passages)} passages from {target_kb}",
        "passages": passages,
        "items": items,
        "tool_reasoning": routing_reason,
        "latency_ms": (time.time() - start_time) * 1000,
    }

    return json.dumps(result_data, ensure_ascii=False)
