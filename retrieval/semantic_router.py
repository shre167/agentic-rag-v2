"""
LLM-Based Semantic Retrieval Router
=====================================
Routes user queries to the most relevant knowledge base using LLM reasoning
over book profiles, NOT by searching every KB.

Flow:
  1. Retrieve registered book profiles (summary, topics, concepts, coverage).
  2. Pass profiles + user query + conversation history to the LLM.
  3. LLM identifies the single most relevant book based on content and coverage.
  4. Returns the corresponding KB ID and collection name.
  5. Returns NO_MATCH if no registered book is relevant.
"""

import json
import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from retrieval.kb_registry import get_all_book_profiles, get_kb_registry
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


def _format_profiles_for_prompt(profiles: List[Dict[str, Any]]) -> str:
    """Format book profiles into a structured text block for the LLM."""
    if not profiles:
        return "No books are currently registered."

    parts = []
    for p in profiles:
        topics = ", ".join(p.get("main_topics", [])[:10]) or "N/A"
        concepts = ", ".join(p.get("key_concepts", [])[:10]) or "N/A"
        parts.append(
            f"- KB ID: `{p['kb_id']}`\n"
            f"  Book: \"{p['kb_name']}\"\n"
            f"  Collection: `{p.get('collection_name', p['kb_id'])}`\n"
            f"  Summary: {p.get('summary', 'N/A')}\n"
            f"  Main Topics: {topics}\n"
            f"  Key Concepts: {concepts}\n"
            f"  Coverage: {p.get('coverage', 'N/A')}\n"
            f"  Chunks: {p.get('total_chunks', 0)}"
        )
    return "\n\n".join(parts)


def route_query(
    query: str,
    chat_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Route a user query to the best matching KB using LLM reasoning over book profiles.

    Returns:
        dict: {
            "is_match": bool,
            "selected_kb": str | None,       # Single KB ID
            "collection_name": str | None,    # Corresponding collection
            "sub_query": str,                 # Refined search query
            "chapter": str | None,
            "page": int | None,
            "line": int | None,
            "reason": str,
        }
    """
    profiles = get_all_book_profiles()
    if not profiles:
        return {
            "is_match": False,
            "selected_kb": None,
            "collection_name": None,
            "sub_query": query,
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "No knowledge bases are currently registered.",
        }

    # Pre-extract direct page/line/chapter hints
    page_match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
    line_match = re.search(r"\bline\s*(\d+)\b", query, re.IGNORECASE)
    extracted_page = int(page_match.group(1)) if page_match else None
    extracted_line = int(line_match.group(1)) if line_match else None

    # Format book profiles and conversation context
    profiles_text = _format_profiles_for_prompt(profiles)
    formatted_history = format_chat_history_for_prompt()

    # Build the LLM prompt
    llm = _get_router_llm()
    if llm is None:
        # If no LLM available, return NO_MATCH — do NOT search all KBs
        return {
            "is_match": False,
            "selected_kb": None,
            "collection_name": None,
            "sub_query": query,
            "chapter": None,
            "page": None,
            "line": None,
            "reason": "Router LLM unavailable. Cannot route without LLM.",
        }

    # Build the valid KB IDs list for validation
    valid_kb_ids = [p["kb_id"] for p in profiles]

    system_instruction = f"""You are the Semantic Router for a book-grounded RAG system.

Your task: Given a user query, select the SINGLE most relevant book to search, or return NO_MATCH.

REGISTERED BOOKS AND THEIR PROFILES:
{profiles_text}

CONVERSATION HISTORY (for context and follow-up resolution):
{formatted_history}

USER QUERY: "{query}"

INSTRUCTIONS:
1. Analyze the query against each book's summary, topics, key concepts, and coverage.
2. Use conversation history to resolve pronouns ("it", "this", "that book") and understand follow-ups.
3. Select the SINGLE most relevant book. Do NOT select multiple books.
4. If the query is clearly outside ALL books' scope (e.g., current news, weather, sports, unrelated trivia), set is_match to false.
5. Extract chapter, page, or line numbers if specifically mentioned in the query.
6. Generate a refined sub_query that removes book-name references and focuses on the actual search intent.

RESPOND ONLY with valid JSON (no markdown, no code blocks):
{{
    "is_match": true,
    "selected_kb": "kb_id_here",
    "sub_query": "refined search query without book name",
    "chapter": null,
    "page": null,
    "line": null,
    "reason": "1 sentence explaining why this book was selected"
}}

For NO_MATCH:
{{
    "is_match": false,
    "selected_kb": null,
    "sub_query": "{query}",
    "chapter": null,
    "page": null,
    "line": null,
    "reason": "1 sentence explaining why no book matches"
}}

Valid KB IDs: {valid_kb_ids}
"""

    try:
        response = llm.invoke(system_instruction)
        raw_text = response.content.strip()

        # Clean markdown code fences
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        is_match = bool(parsed.get("is_match", False))
        selected_kb = parsed.get("selected_kb")

        # Validate the selected KB exists
        registry = get_kb_registry()
        if selected_kb and selected_kb not in registry:
            # Try to match by partial name
            selected_kb = None
            is_match = False

        # Get collection name
        collection_name = None
        if selected_kb and selected_kb in registry:
            collection_name = registry[selected_kb].get("collection_name", selected_kb)
        else:
            is_match = False
            selected_kb = None

        sub_query = parsed.get("sub_query", query)
        if not isinstance(sub_query, str) or not sub_query.strip():
            sub_query = query

        return {
            "is_match": is_match,
            "selected_kb": selected_kb,
            "collection_name": collection_name,
            "sub_query": sub_query,
            "chapter": parsed.get("chapter"),
            "page": parsed.get("page") if parsed.get("page") is not None else extracted_page,
            "line": parsed.get("line") if parsed.get("line") is not None else extracted_line,
            "reason": parsed.get("reason", "Routed based on semantic analysis."),
        }

    except Exception as exc:
        # On LLM failure, return NO_MATCH — do NOT fall back to searching all KBs
        return {
            "is_match": False,
            "selected_kb": None,
            "collection_name": None,
            "sub_query": query,
            "chapter": None,
            "page": None,
            "line": None,
            "reason": f"Router failed: {str(exc)[:100]}. No fallback to all-KB search.",
        }
