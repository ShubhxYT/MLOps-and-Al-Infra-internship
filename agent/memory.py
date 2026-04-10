import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from .models import Intent, ToolResult

_HISTORY_FILE = Path("output/session_history.json")
_SESSION_KEY = "memo_history"


class HistoryEntry:
    """Immutable snapshot of one complete pipeline run."""

    def __init__(
        self,
        transcription: str,
        intents: list[Intent],
        results: list[ToolResult],
        audio_source: str = "unknown",
    ) -> None:
        self.timestamp = datetime.now().isoformat()
        self.audio_source = audio_source
        self.transcription = transcription
        # Store as plain dicts for JSON-safe serialization
        self.intents = [i.model_dump() for i in intents]
        self.results = [r.model_dump() for r in results]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "audio_source": self.audio_source,
            "transcription": self.transcription,
            "intents": self.intents,
            "results": self.results,
        }


class SessionMemory:
    """Manages session history backed by st.session_state with optional JSON persistence."""

    @staticmethod
    def _ensure() -> None:
        if _SESSION_KEY not in st.session_state:
            st.session_state[_SESSION_KEY] = []

    @staticmethod
    def add(entry: HistoryEntry) -> None:
        """Append a new history entry and persist to disk."""
        SessionMemory._ensure()
        st.session_state[_SESSION_KEY].append(entry.to_dict())
        SessionMemory._persist()

    @staticmethod
    def get_all() -> list[dict[str, Any]]:
        """Return all history entries in chronological order."""
        SessionMemory._ensure()
        return st.session_state[_SESSION_KEY]

    @staticmethod
    def get_chat_history() -> list[dict[str, str]]:
        """Return simplified {user, assistant} pairs for LLM context injection."""
        entries = SessionMemory.get_all()
        history: list[dict[str, str]] = []
        for entry in entries:
            user_msg = entry.get("transcription", "")
            # Collect successful outputs as the "assistant" side
            assistant_parts = [
                r["output"]
                for r in entry.get("results", [])
                if r.get("success") and r.get("output")
            ]
            if user_msg and assistant_parts:
                history.append(
                    {"user": user_msg, "assistant": " | ".join(assistant_parts)}
                )
        return history

    @staticmethod
    def clear() -> None:
        """Wipe session history from state and disk."""
        st.session_state[_SESSION_KEY] = []
        if _HISTORY_FILE.exists():
            _HISTORY_FILE.unlink()

    @staticmethod
    def _persist() -> None:
        """Write current history to JSON file (non-critical — failures are silently ignored)."""
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = SessionMemory.get_all()
            _HISTORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
