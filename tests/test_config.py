"""Tests for configuration system."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from signalrag.core.config import ChunkingConfig, RetrievalConfig, Settings


def test_default_settings():
    settings = Settings()
    assert settings.app_name == "SignalRAG"
    assert settings.chunking.chunk_size == 512
    assert settings.chunking.chunk_overlap == 64
    assert settings.retrieval.top_k == 5
    assert settings.retrieval.hybrid_alpha == 0.5


def test_chunking_overlap_validation():
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=100, chunk_overlap=150)


def test_retrieval_alpha_validation():
    with pytest.raises(ValidationError):
        RetrievalConfig(hybrid_alpha=1.5)

    with pytest.raises(ValidationError):
        RetrievalConfig(hybrid_alpha=-0.1)


def test_yaml_export_and_load(tmp_path: Path):
    settings = Settings()
    settings.chunking.chunk_size = 256
    settings.chunking.chunk_overlap = 32

    yaml_file = tmp_path / "custom_config.yaml"
    settings.to_yaml(yaml_file)

    loaded = Settings.from_yaml(yaml_file)
    assert loaded.chunking.chunk_size == 256
    assert loaded.chunking.chunk_overlap == 32


def test_yaml_load_default():
    default_yaml = Path("configs/default.yaml")
    assert default_yaml.is_file()
    settings = Settings.from_yaml(default_yaml)
    assert settings.app_name == "SignalRAG"
    assert settings.embedding.dimension == 384
