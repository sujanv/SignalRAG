"""Tests for demo script execution."""

from scripts.demo import main


def test_demo_execution(capsys):
    main()
    captured = capsys.readouterr()
    assert "SignalRAG Portfolio Demonstration" in captured.out
    assert "Indexed 3 documents" in captured.out
    assert "Demo completed successfully!" in captured.out
