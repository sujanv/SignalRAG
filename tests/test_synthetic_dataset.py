"""Tests for synthetic dataset generator."""

import tempfile
from pathlib import Path

from signalrag.evaluation.dataset import EvaluationDataset
from signalrag.evaluation.synthetic import SyntheticDatasetGenerator
from signalrag.models.chunk import Chunk, ChunkMetadata


def test_synthetic_dataset_generation():
    chunks = [
        Chunk(
            id="chunk_1",
            document_id="doc_rag",
            text="SignalRAG uses BM25 retrieval for high-precision keyword searching in corpora.",
            metadata=ChunkMetadata(document_id="doc_rag", chunk_index=0, source="rag.pdf"),
        ),
        Chunk(
            id="chunk_2",
            document_id="doc_vector",
            text="Vector embeddings capture semantic similarity across high-dimensional semantic spaces.",
            metadata=ChunkMetadata(document_id="doc_vector", chunk_index=1, source="vector.pdf"),
        ),
    ]

    generator = SyntheticDatasetGenerator()
    dataset = generator.generate_from_chunks(chunks, max_examples=2)

    assert len(dataset) == 2
    assert dataset[0].id == "synth_0001"
    assert "doc_rag" in dataset[0].expected_doc_ids
    assert "chunk_1" in dataset[0].expected_chunk_ids
    assert len(dataset[0].question) > 5
    assert len(dataset[0].ground_truth_answer) > 5

    # Check disk serialization and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "synthetic.json"
        dataset.to_json(out_file)
        loaded = EvaluationDataset.from_json(out_file)
        assert len(loaded) == 2
        assert loaded[0].id == dataset[0].id
