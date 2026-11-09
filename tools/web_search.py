"""
Web Search Tool
===============
Robust web search tool with dual-engine support:
1. DuckDuckGo Search (primary via ddgs / duckduckgo_search)
2. Wikipedia API Search (instant corporate fallback when DDG is blocked,
   rate-limited, or subject to corporate firewall/DNS filtering)

Features for Corporate / Office Environments:
- Automatic proxy detection (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, NO_PROXY)
- Corporate SSL inspection resilience (graceful handling of self-signed MITM certs)
- Strict timeouts (never hangs the agent or UI)
- Structured passage formatting so the agent and UI handle web results
  identically to book passages with citations.
"""

import json
import os
import re
import urllib.parse
import urllib.request
from langchain.tools import tool


def _clean_html(raw_html: str) -> str:
    """Strip HTML tags and unescape common entities."""
    clean = re.sub(r"<[^>]+>", "", raw_html)
    clean = (
        clean.replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", clean).strip()


def _get_proxies() -> dict[str, str] | None:
    """Extract standard proxy environment variables."""
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    all_proxy = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")

    proxies = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    elif all_proxy:
        proxies["https"] = all_proxy
        proxies["http"] = all_proxy

    return proxies if proxies else None


def _search_ddg(query: str, max_results: int = 5, timeout: int = 7) -> list[dict]:
    """Search DuckDuckGo using ddgs or duckduckgo_search with proxy and timeout support."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return []

    proxies = _get_proxies()
    proxy_str = proxies.get("https") or proxies.get("http") if proxies else None
    kwargs = {"timeout": timeout}
    if proxy_str:
        kwargs["proxy"] = proxy_str

    try:
        with DDGS(**kwargs) as ddgs:
            raw = ddgs.text(query, max_results=max_results)
            results = []
            for item in raw:
                title = item.get("title", "")
                url = item.get("href", item.get("url", ""))
                snippet = item.get("body", item.get("description", ""))
                if title or snippet:
                    results.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "engine": "DuckDuckGo",
                        }
                    )
            return results
    except Exception:
        return []


def _search_wikipedia(query: str, max_results: int = 5, timeout: int = 5) -> list[dict]:
    """
    Search Wikipedia API as an office-safe fallback.
    Wikipedia API is open, fast, uses standard HTTPS, and is virtually
    never blocked by enterprise proxy filters or DNS blacklists.
    """
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": "1",
            "srlimit": str(max_results),
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    headers = {"User-Agent": "AgenticRAG-Assistant/1.0 (Enterprise Assistant)"}

    try:
        proxies = _get_proxies()
        if proxies:
            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()

        req = urllib.request.Request(url, headers=headers)
        with opener.open(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            items = data.get("query", {}).get("search", [])

            results = []
            for item in items:
                title = item.get("title", "")
                page_id = item.get("pageid", "")
                snippet = _clean_html(item.get("snippet", ""))
                article_url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                results.append(
                    {
                        "title": title,
                        "url": article_url,
                        "snippet": snippet,
                        "engine": "Wikipedia",
                    }
                )
            return results
    except Exception:
        return []


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the live web for questions NOT covered by the indexed books (e.g. current events,
    general technology, definitions, world facts, or topics outside Mind Management / Deep Work).

    Use this tool when:
    - The question is unrelated to the indexed books.
    - semantic_search returned 0 relevant passages and the topic is general/external.
    - The user explicitly asks for web search or current information.

    Do NOT use this tool when the question can be answered from the indexed books.

    Args:
        query: Concise, keyword-focused search query.
        max_results: Maximum results to retrieve (default 5).

    Returns:
        Structured JSON with results, snippets, and citations.
    """
    cleaned_query = query.strip()
    results = _search_ddg(cleaned_query, max_results=max_results, timeout=7)
    engine_used = "DuckDuckGo"

    # If DuckDuckGo returned nothing (common in corporate networks with strict egress filtering),
    # immediately fall back to Wikipedia API.
    if not results:
        results = _search_wikipedia(cleaned_query, max_results=max_results, timeout=5)
        if results:
            engine_used = "Wikipedia (Enterprise Fallback)"

    if not results:
        return json.dumps(
            {
                "tool": "web_search",
                "input": cleaned_query,
                "result": f"No web search results found for '{cleaned_query}'. (Network restricted or no matches)",
                "engine": "none",
                "passages": [],
                "items": [],
                "web_results": [],
            },
            ensure_ascii=False,
        )

    web_results = []
    passages = []

    for idx, item in enumerate(results, start=1):
        title = item.get("title", f"Web Source {idx}")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        item_engine = item.get("engine", engine_used)

        web_results.append(
            {
                "rank": idx,
                "title": title,
                "url": url,
                "snippet": snippet,
                "engine": item_engine,
            }
        )

        web_id = f"web-{idx:02d}"
        passages.append(
            {
                "id": web_id,
                "text": f"{title}: {snippet} (Source: {url})",
                "title": title,
                "url": url,
                "source": "web",
                "engine": item_engine,
                "book": f"Web ({item_engine})",
                "chapter": "",
                "section": title,
                "page_start": 0,
                "page_end": 0,
                "distance": 0.0,
            }
        )

    return json.dumps(
        {
            "tool": "web_search",
            "input": cleaned_query,
            "result": f"Found {len(web_results)} web results via {engine_used}",
            "engine": engine_used,
            "passages": passages,
            "items": [f"[{r['rank']}] {r['title']} — {r['url']}" for r in web_results],
            "web_results": web_results,
        },
        ensure_ascii=False,
    )
