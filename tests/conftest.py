"""Pytest shared fixtures."""

import pytest


@pytest.fixture
def sample_text() -> str:
    return "SignalRAG provides production-grade retrieval augmented generation and evaluation."
