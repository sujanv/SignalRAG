"""Sanity tests for SignalRAG."""

from signalrag import __version__


def test_version():
    assert __version__ == "0.1.0"


def test_sample_fixture(sample_text: str):
    assert "SignalRAG" in sample_text
