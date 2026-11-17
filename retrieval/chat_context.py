"""
Thread-safe conversational context tracker.
Stores previous conversation turns as context for the semantic retrieval router and agent.
"""
import contextvars
from typing import List, Dict, Any

_current_chat_context = contextvars.ContextVar("current_chat_context", default=[])


def set_chat_context(history: List[Dict[str, Any]] | None) -> None:
    """Set the conversation history for the current request thread/context."""
    _current_chat_context.set(list(history or []))


def get_chat_context() -> List[Dict[str, Any]]:
    """Retrieve the conversation history for the current context."""
    return _current_chat_context.get()


def format_chat_history_for_prompt(max_turns: int = 6) -> str:
    """Format recent conversation turns into a clean text block."""
    history = get_chat_context()
    if not history:
        return "No prior conversation turns."

    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        role = str(turn.get("role", "user")).capitalize()
        content = str(turn.get("content", "")).strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)
