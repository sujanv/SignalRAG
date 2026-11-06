"""Agentic conversational memory and multi-turn dialogue state management for RAG."""

from __future__ import annotations

import re
import time
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single message in a conversational session."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = Field(default_factory=time.time)


class ConversationSession(BaseModel):
    """Stateful dialogue history container with sliding-window truncation."""

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    max_history_turns: int = 10

    def add_message(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))
        # Keep within max turns (each turn has user + assistant, so max_turns * 2)
        if len(self.messages) > self.max_history_turns * 2:
            self.messages = self.messages[-self.max_history_turns * 2 :]


class ConversationalRAGManager:
    """Manages multi-turn conversation sessions, query condensation, and contextual coreference."""

    def __init__(self):
        self.sessions: dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]

    def add_turn(self, session_id: str, user_query: str, assistant_reply: str) -> None:
        session = self.get_or_create_session(session_id)
        session.add_message("user", user_query)
        session.add_message("assistant", assistant_reply)

    def condense_query(self, session_id: str, current_query: str) -> str:
        """Resolve pronouns and coreference in follow-up queries using conversation history."""
        session = self.get_or_create_session(session_id)
        if not session.messages:
            return current_query

        # Check for demonstrative or coreferent pronouns
        pronoun_match = re.search(
            r"\b(it|its|this|that|these|those|they|their)\b", current_query, re.IGNORECASE
        )
        if not pronoun_match:
            return current_query

        # Find the last user query to extract the main subject
        last_user_query = ""
        for msg in reversed(session.messages):
            if msg.role == "user":
                last_user_query = msg.content
                break

        if not last_user_query:
            return current_query

        # Extract primary nouns / capitalized terms or prominent words from last query
        candidates = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", last_user_query)
        stop_words = {
            "how",
            "what",
            "why",
            "when",
            "where",
            "who",
            "which",
            "can",
            "does",
            "is",
            "are",
            "the",
        }
        filtered_candidates = [c for c in candidates if c.lower() not in stop_words]
        if filtered_candidates:
            subject = filtered_candidates[0]
        else:
            words = [
                w
                for w in re.findall(r"\w+", last_user_query)
                if len(w) > 3 and w.lower() not in stop_words
            ]
            subject = words[-1] if words else "the system"

        # Condense query replacing pronoun
        condensed = re.sub(
            r"\b(it|this|that)\b",
            subject,
            current_query,
            flags=re.IGNORECASE,
        )
        return condensed

    def format_history_context(self, session_id: str) -> str:
        """Format recent dialogue history into context block for LLM prompt."""
        session = self.get_or_create_session(session_id)
        if not session.messages:
            return ""

        formatted_lines = ["[Conversation History]"]
        for msg in session.messages[-6:]:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted_lines.append(f"{role_label}: {msg.content}")
        return "\n".join(formatted_lines)
