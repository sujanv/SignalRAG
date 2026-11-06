"""Tests for Conversational RAG Memory & Coreference Resolution."""

from signalrag.generation.conversational import ConversationalRAGManager


def test_conversational_manager_coreference_resolution():
    mgr = ConversationalRAGManager()
    session_id = "test_user_session"

    mgr.add_turn(
        session_id,
        "How does GraphRAG operate?",
        "GraphRAG indexes entities and relationship edges.",
    )

    # Follow-up query with pronoun 'it'
    condensed = mgr.condense_query(session_id, "Can it handle circular graph dependencies?")
    assert "GraphRAG" in condensed

    history_str = mgr.format_history_context(session_id)
    assert "[Conversation History]" in history_str
    assert "User: How does GraphRAG operate?" in history_str
    assert "Assistant: GraphRAG indexes entities" in history_str


def test_conversational_manager_sliding_window():
    mgr = ConversationalRAGManager()
    session = mgr.get_or_create_session("sess_window")
    session.max_history_turns = 2

    for i in range(10):
        mgr.add_turn("sess_window", f"Query {i}", f"Answer {i}")

    # Should retain only max_history_turns * 2 = 4 messages
    assert len(session.messages) <= 4
    assert session.messages[-1].content == "Answer 9"
