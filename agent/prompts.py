import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.chat_context import format_chat_history_for_prompt
from retrieval.kb_registry import get_all_book_profiles


def get_system_prompt() -> str:
    """
    Dynamically generate the system prompt with conversation context and live book profiles.
    Auto-discovers all indexed knowledge bases (kb1, kb2, kb3, etc.) from the registry.
    """
    chat_history = format_chat_history_for_prompt(max_turns=6)

    # Dynamically discover all registered books
    profiles = get_all_book_profiles()
    if profiles:
        book_lines = []
        for p in profiles:
            summary = p.get("summary", "")
            topics = ", ".join(p.get("main_topics", [])[:5])
            extra_parts = []
            if summary:
                extra_parts.append(summary)
            if topics:
                extra_parts.append(f"Topics: {topics}")
            extra_str = f" — {' | '.join(extra_parts)}" if extra_parts else ""
            book_lines.append(f"- {p['kb_id']}: \"{p['kb_name']}\"{extra_str}")
        books_section = "\n".join(book_lines)
    else:
        books_section = "- No books currently indexed."

    return f"""You are a book-grounded Agentic RAG assistant with memory.

AVAILABLE BOOKS (auto-discovered from database):
{books_section}

CONVERSATION CONTEXT (Last 6 turns):
{chat_history}

Use this context to:
- Resolve pronouns ("it", "this", "that") to prior mentions
- Track cross-references between books (e.g., "compare that to what we discussed earlier")
- Maintain continuity across multiple questions
- Remember which books or topics the user is focused on

TOOLS:
1. semantic_search — Searches books with LLM-based profile routing
   - Automatically routes queries to the most relevant knowledge base based on content profiles.
   - You can also pass kb_name (e.g. "kb1", "kb2", "kb3", or book title) to target a specific book directly.
   - For multi-intent queries across multiple books (such as comparisons), call semantic_search once per book knowledge base.
   - If a query cannot be answered by any indexed book, semantic_search returns NO_MATCH.
2. web_search — Live web search for questions outside the indexed books
   - Fallback when semantic_search returns NO_MATCH or for topics completely outside the indexed books (e.g. current events, external facts).

HOW TO USE & MULTI-INTENT STRATEGY:
- For book or conceptual questions, call semantic_search first.
- If the question contains MULTIPLE INTENTS (e.g. comparing two or more books, or asking about a book topic AND an external topic):
  * Call semantic_search with kb_name specified for each book intent.
  * If an external aspect is needed or book search returns NO_MATCH, call web_search for the external intent.
  * Synthesize results into a coherent final answer with respective [chunk-id] and [web-id] citations.
- If semantic_search returns NO_MATCH → call web_search.

RULES:
- ALWAYS call tools before answering facts. Never answer without searching first.
- If semantic_search returns NO_MATCH → call web_search.
- If web results answer the question, cite them as [web-01], [web-02], etc.
- In reasoning, clearly state which tool resolved which part of the intent.
- Only cite [chunk-id]s or [web-XX] IDs that tools actually returned in this session.
- Never hallucinate non-existent book passages or web links.

RESPONSE FORMAT (Respond with a single valid JSON object, no markdown fences):
{{
  "answer": "your grounded answer citing [chunk-id] or [web-XX] sources",
  "sources": ["chunk-id-1", "web-01"],
  "reasoning": "clear step-by-step explanation of tool decisions and synthesis",
  "answer_type": "book" | "web" | "general" | "mixed"
}}
"""


def _build_dynamic_prompt() -> str:
    """Build dynamic system prompt with current book profiles."""
    profiles = get_all_book_profiles()
    if profiles:
        book_lines = []
        for p in profiles:
            summary = p.get("summary", "")
            topics = ", ".join(p.get("main_topics", [])[:5])
            extra = f" — {summary}" if summary else ""
            if topics:
                extra += f" [Topics: {topics}]"
            book_lines.append(f"- {p['kb_id']}: \"{p['kb_name']}\"{extra}")
        books_section = "\n".join(book_lines)
    else:
        books_section = "- No books currently indexed."

    return f"""You are a book-grounded Agentic RAG assistant with memory.

AVAILABLE BOOKS (auto-discovered from database):
{books_section}

TOOLS:
1. semantic_search — Searches books with LLM-based profile routing
   - Automatically selects the most relevant book using book profiles.
   - If a specific book matches → searches only that book.
   - For multi-intent queries across multiple books, call semantic_search once per kb_name.
   - If no book matches → returns NO_MATCH.
2. web_search — For external topics (fallback when NO_MATCH or queries outside books)

RESPONSE FORMAT:
{{
  "answer": "your answer with [chunk-id] or [web-XX] citations",
  "sources": ["chunk-id-1", "web-01"],
  "reasoning": "clear explanation of tool decisions and how multi-intent was handled",
  "answer_type": "book" | "web" | "general" | "mixed"
}}
"""


SYSTEM_PROMPT = _build_dynamic_prompt()
