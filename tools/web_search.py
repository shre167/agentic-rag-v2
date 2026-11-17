"""Web search tool - ONLY use as fallback when book search fails."""

import json
import time
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from tools.results import tool_result


@tool
def web_search(query: str) -> str:
    """
    Search the web for external information.
    
    USE ONLY AS FALLBACK when:
    - Book search (semantic_search) returned no/insufficient results
    - Query is about external topics not in books (current events, general knowledge)
    
    DO NOT use for book-related topics.
    """
    start_time = time.time()
    
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)
        
        if not results:
            return tool_result(
                "web_search",
                query,
                "No web results found.",
                passages=[],
                items=[],
                latency_ms=(time.time() - start_time) * 1000,
            )
        
        # Convert web results to passage format for consistency
        web_passages = [
            {
                "id": f"web-{i:02d}",
                "text": r.get('body', 'No description'),
                "title": r.get('title', 'No title'),
                "url": r.get('href', ''),
                "source": "web",
                "book": "Web Result",
                "chapter": "",
                "section": r.get('title', 'Web Result'),
                "page_start": 0,
                "page_end": 0,
            }
            for i, r in enumerate(results, 1)
        ]
        
        # Format summary
        result_summary = f"Found {len(web_passages)} web result(s)"
        
        return tool_result(
            "web_search",
            query,
            result_summary,
            passages=web_passages,
            items=[f"[web-{i:02d}] {r.get('title', 'Result')}" for i, r in enumerate(results, 1)],
            latency_ms=(time.time() - start_time) * 1000,
        )
        
    except Exception as e:
        return tool_result(
            "web_search",
            query,
            f"Web search error: {str(e)}",
            passages=[],
            items=[],
            latency_ms=(time.time() - start_time) * 1000,
        )