"""
Persistent chat history storage using JSON files.
Survives page refreshes and browser restarts.
"""

import json
from pathlib import Path
from datetime import datetime
from settings import ROOT


CHAT_HISTORY_DIR = ROOT / "chat_history"
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# Single session file (default)
DEFAULT_SESSION_FILE = CHAT_HISTORY_DIR / "default_session.json"


def load_chat_history(session_id: str = "default") -> list[dict]:
    """Load chat history from persistent storage."""
    session_file = CHAT_HISTORY_DIR / f"{session_id}_session.json"
    
    if not session_file.exists():
        return []
    
    try:
        with open(session_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_chat_history(messages: list[dict], session_id: str = "default") -> None:
    """Save chat history to persistent storage."""
    session_file = CHAT_HISTORY_DIR / f"{session_id}_session.json"
    
    try:
        with open(session_file, "w") as f:
            json.dump(messages, f, indent=2)
    except IOError as e:
        print(f"Failed to save chat history: {e}")


def clear_chat_history(session_id: str = "default") -> None:
    """Clear chat history for a session."""
    session_file = CHAT_HISTORY_DIR / f"{session_id}_session.json"
    
    if session_file.exists():
        session_file.unlink()


def get_all_sessions() -> list[str]:
    """Get list of all session IDs."""
    sessions = [
        f.stem.replace("_session", "")
        for f in CHAT_HISTORY_DIR.glob("*_session.json")
    ]
    if "default" not in sessions:
        sessions.insert(0, "default")
    return sorted(list(set(sessions)))


def get_cross_session_summary(current_session_id: str = "default", max_turns_per_session: int = 2) -> str:
    """
    Summarize previous queries from other sessions so the LLM remembers
    what topics/queries were explored previously across sessions.
    """
    all_sessions = get_all_sessions()
    other_sessions = [s for s in all_sessions if s != current_session_id]
    if not other_sessions:
        return ""

    summary_lines = []
    for s_id in other_sessions:
        history = load_chat_history(s_id)
        user_queries = [
            turn["content"]
            for turn in history
            if turn.get("role") == "user" and turn.get("content")
        ]
        if user_queries:
            recent_queries = user_queries[-max_turns_per_session:]
            queries_str = "; ".join(f'"{q}"' for q in recent_queries)
            summary_lines.append(f"- Session '{s_id}': {queries_str}")

    if not summary_lines:
        return ""

    return "Past topics/queries explored in other sessions:\n" + "\n".join(summary_lines)
