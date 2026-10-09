"""Tests for GitHub Actions CI workflow configuration."""

from pathlib import Path

import yaml


def test_ci_workflow_valid():
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.exists()
    content = ci_path.read_text()
    data = yaml.safe_load(content)

    assert "jobs" in data
    assert "test" in data["jobs"]
    steps = data["jobs"]["test"]["steps"]
    step_names = [s.get("name", "") for s in steps]

    assert any("Checkout" in n for n in step_names)
    assert any("Lint check" in n for n in step_names)
    assert any("Test Suite" in n for n in step_names)
    assert any("Build" in n for n in step_names)
