"""
Thread-safe conversational context tracker.
Stores previous conversation turns as context for the semantic retrieval router and agent.
"""
import contextvars
from typing import List, Dict, Any

_current_chat_context = contextvars.ContextVar("current_chat_context", default=[])
_cross_session_context = contextvars.ContextVar("cross_session_context", default="")


def set_chat_context(history: List[Dict[str, Any]] | None, cross_session_info: str = "") -> None:
    """Set the conversation history and optional cross-session info for current context."""
    _current_chat_context.set(list(history or []))
    _cross_session_context.set(cross_session_info or "")


def get_chat_context() -> List[Dict[str, Any]]:
    """Retrieve the conversation history for the current context."""
    return _current_chat_context.get()


def format_chat_history_for_prompt(max_turns: int = 6) -> str:
    """Format recent conversation turns and cross-session memory into a clean text block."""
    history = get_chat_context()
    cross_info = _cross_session_context.get()

    lines = []
    if cross_info:
        lines.append(f"[Cross-Session Memory]\n{cross_info}\n")

    if not history:
        if not lines:
            return "No prior conversation turns."
        return "\n".join(lines)

    recent = history[-max_turns:]
    lines.append("[Current Session Context]")
    for turn in recent:
        role = str(turn.get("role", "user")).capitalize()
        content = str(turn.get("content", "")).strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)
