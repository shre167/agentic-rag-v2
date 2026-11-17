"""
Semantic Retrieval Router
==========================
Uses a two-stage hybrid approach:
  1. Embeddings: Rapidly ranks and scores all ingested knowledge bases against the query.
  2. LLM Decision: Reasons over candidates + conversation turns (chat history) to select
     the best matching KB(s), resolve pronouns, extract metadata filters, or detect if the
     query is irrelevant to indexed books.
"""

import json
import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from retrieval.kb_registry import find_kb_candidates, get_kb_registry
from retrieval.chat_context import format_chat_history_for_prompt
from settings import ROOT

load_dotenv(ROOT / ".env")

_router_llm: ChatOpenAI | None = None


def _get_router_llm() -> ChatOpenAI | None:
    """Initialize or reuse LLM client for routing decisions."""
    global _router_llm
    if _router_llm is None:
        api_key = os.getenv("GE_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        model_name = os.getenv("LLM_MODEL")

        if api_key and base_url and model_name:
            _router_llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
            )
    return _router_llm


def route_query(
    query: str,
    chat_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Route a user query using embeddings + LLM reasoning over chat history.

    Returns:
        dict: {
            "is_match": bool,
            "selected_kbs": list[str],
            "sub_queries": dict[str, str],
            "chapter": str | None,
            "page": int | None,
            "line": int | None,
            "reason": str,
        }
    """
    candidates = find_kb_candidates(query)
    if not candidates:
        return {
            "is_match": False,
            "selected_kbs": [],
            "sub_queries": {},
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "No knowledge bases are currently indexed in ChromaDB.",
        }

    # Format candidate summary for LLM prompt
    kb_summaries = []
    for c in candidates:
        chapters_str = ", ".join(c["chapters"]) if c["chapters"] else "General"
        kb_summaries.append(
            f"- `{c['kb_name']}`: \"{c['book_title']}\" (Embedding similarity: {c['similarity']:.2f}, Chapters: {chapters_str})"
        )
    catalog_text = "\n".join(kb_summaries)

    # Format recent conversation context
    formatted_history = format_chat_history_for_prompt()

    # Pre-extract direct page/line/chapter hints if present in query
    page_match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
    line_match = re.search(r"\bline\s*(\d+)\b", query, re.IGNORECASE)
    extracted_page = int(page_match.group(1)) if page_match else None
    extracted_line = int(line_match.group(1)) if line_match else None

    # ─────────────────────────────────────────────────────────────
    # Fast-Path 1: Direct Title / Author Keyword Match (0 ms)
    # ─────────────────────────────────────────────────────────────
    q_lower = query.lower()
    for c in candidates:
        title_words = [w for w in c["book_title"].lower().split() if len(w) > 3]
        if c["book_title"].lower() in q_lower or (title_words and all(w in q_lower for w in title_words)):
            return {
                "is_match": True,
                "selected_kbs": [c["kb_name"]],
                "sub_queries": {c["kb_name"]: query},
                "chapter": None,
                "page": extracted_page,
                "line": extracted_line,
                "reason": f"Direct title match: '{c['book_title']}'",
            }

    top = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None

    # ─────────────────────────────────────────────────────────────
    # Fast-Path 2: Decisive Embedding Similarity Match (< 2 ms)
    # Avoids redundant LLM round-trip when similarity is confident.
    # ─────────────────────────────────────────────────────────────
    is_comparative = bool(re.search(r"\b(compare|both|difference|versus|vs)\b", q_lower))

    if is_comparative and runner_up and top["similarity"] >= 0.22 and runner_up["similarity"] >= 0.22:
        selected = [top["kb_name"], runner_up["kb_name"]]
        return {
            "is_match": True,
            "selected_kbs": selected,
            "sub_queries": {kb: query for kb in selected},
            "chapter": None,
            "page": extracted_page,
            "line": extracted_line,
            "reason": f"Comparative match across {top['book_title']} ({top['similarity']:.2f}) and {runner_up['book_title']} ({runner_up['similarity']:.2f})",
        }

    if top["similarity"] >= 0.25 and (runner_up is None or (top["similarity"] - runner_up["similarity"]) >= 0.05):
        return {
            "is_match": True,
            "selected_kbs": [top["kb_name"]],
            "sub_queries": {top["kb_name"]: query},
            "chapter": None,
            "page": extracted_page,
            "line": extracted_line,
            "reason": f"High confidence embedding match with {top['book_title']} ({top['similarity']:.2f})",
        }

    # ─────────────────────────────────────────────────────────────
    # Fast-Path 3: Irrelevant Query Check (< 0.10 similarity)
    # ─────────────────────────────────────────────────────────────
    if top["similarity"] < 0.10:
        return {
            "is_match": False,
            "selected_kbs": [],
            "sub_queries": {},
            "chapter": None,
            "page": None,
            "line": None,
            "reason": f"Embedding similarity too low ({top['similarity']:.2f}) — query is outside indexed books.",
        }

    # ─────────────────────────────────────────────────────────────
    # Ambiguous / Boundary Case: Use Router LLM as Tiebreaker
    # ─────────────────────────────────────────────────────────────
    llm = _get_router_llm()
    if llm is None:
        # Fallback to top embedding candidate
        return {
            "is_match": True,
            "selected_kbs": [top["kb_name"]],
            "sub_queries": {top["kb_name"]: query},
            "chapter": None,
            "page": extracted_page,
            "line": extracted_line,
            "reason": f"Fallback to {top['book_title']} by embedding similarity ({top['similarity']:.2f}).",
        }

    system_instruction = f"""You are the Semantic Retrieval Router for an Agentic RAG system with multi-book collections.

Available Knowledge Bases (ranked by embedding relevance):
{catalog_text}

Recent Conversation Context (Chat History):
{formatted_history}

User Query: "{query}"

Instructions:
1. Determine `is_match`:
   - Set to `true` if the query is asking about topics, concepts, chapters, pages, or authors covered in ANY indexed book, OR if it refers to topics discussed in the chat history.
   - Set to `false` if the query is clearly outside the scope of all indexed books (e.g. current live news, stock prices, weather, sports scores, or unrelated trivia).
2. Select `selected_kbs`:
   - List the relevant kb_name(s).
   - If the query compares multiple books (e.g. "compare focus in both books"), select ALL relevant KBs (e.g. ["kb1", "kb2"]).
   - If `is_match` is false, return `[]`.
3. Provide `sub_queries`:
   - A mapping of each selected kb_name to a refined search query.
   - Resolve any conversational pronouns (e.g., replace "he", "that concept", "the author" using the chat history).
4. Extract `chapter`, `page`, `line` if specifically requested.
5. Provide a 1-sentence `reason` explaining the routing decision.

Respond ONLY with valid JSON (no markdown formatting, no code blocks):
{{
  "is_match": true,
  "selected_kbs": ["kb1"],
  "sub_queries": {{"kb1": "search term"}},
  "chapter": null,
  "page": null,
  "line": null,
  "reason": "Why this KB was selected"
}}
"""

    try:
        response = llm.invoke(system_instruction)
        raw_text = response.content.strip()

        # Clean markdown code fences if model enclosed JSON
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        is_match = bool(parsed.get("is_match", True))
        selected_kbs = [
            str(kb).strip()
            for kb in parsed.get("selected_kbs", [])
            if str(kb).strip() in get_kb_registry()
        ]

        if not selected_kbs:
            is_match = False

        sub_queries = parsed.get("sub_queries")
        if not isinstance(sub_queries, dict):
            sub_queries = {kb: query for kb in selected_kbs}
        else:
            for kb in selected_kbs:
                if kb not in sub_queries or not sub_queries[kb]:
                    sub_queries[kb] = query

        return {
            "is_match": is_match,
            "selected_kbs": selected_kbs,
            "sub_queries": sub_queries,
            "chapter": parsed.get("chapter"),
            "page": parsed.get("page") if parsed.get("page") is not None else extracted_page,
            "line": parsed.get("line") if parsed.get("line") is not None else extracted_line,
            "reason": parsed.get("reason", f"Routed to {selected_kbs} based on semantic analysis."),
        }

    except Exception as exc:
        # Graceful fallback: check if top candidate has high embedding similarity
        top = candidates[0]
        if top["similarity"] >= 0.20:
            return {
                "is_match": True,
                "selected_kbs": [top["kb_name"]],
                "sub_queries": {top["kb_name"]: query},
                "chapter": None,
                "page": extracted_page,
                "line": extracted_line,
                "reason": f"Fallback to {top['book_title']} (embedding similarity {top['similarity']:.2f}).",
            }
        return {
            "is_match": False,
            "selected_kbs": [],
            "sub_queries": {},
            "chapter": None,
            "page": None,
            "line": None,
            "reason": f"Query did not match indexed books (fallback).",
        }
