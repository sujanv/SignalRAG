"""Tests for Dockerfile and docker-compose configurations."""

from pathlib import Path

import yaml


def test_dockerfile_present():
    df = Path("Dockerfile")
    assert df.exists()
    content = df.read_text()
    assert "FROM python:3.11-slim" in content
    assert "HEALTHCHECK" in content
    assert "signalrag" in content
    assert "EXPOSE 8000" in content


def test_docker_compose_valid():
    compose_file = Path("docker-compose.yml")
    assert compose_file.exists()
    data = yaml.safe_load(compose_file.read_text())
    assert "services" in data
    assert "signalrag" in data["services"]
    svc = data["services"]["signalrag"]
    assert "8000:8000" in svc["ports"]
    assert "healthcheck" in svc


def test_dockerignore_present():
    di = Path(".dockerignore")
    assert di.exists()
    content = di.read_text()
    assert ".git" in content
    assert ".venv" in content
