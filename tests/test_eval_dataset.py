"""Tests for evaluation dataset schema, loading, and manipulation."""

import json
from pathlib import Path

import pytest

from signalrag.evaluation.dataset import EvalExample, EvaluationDataset


def test_eval_example_model():
    example = EvalExample(
        id="test-1",
        question="What is RRF?",
        ground_truth_answer="Reciprocal Rank Fusion",
        expected_doc_ids=["doc-1"],
        expected_chunk_ids=["chunk-1"],
        metadata={"category": "retrieval"},
    )
    assert example.id == "test-1"
    assert example.expected_chunk_ids == ["chunk-1"]
    assert example.metadata["category"] == "retrieval"


def test_evaluation_dataset_load(tmp_path: Path):
    file_path = tmp_path / "test_benchmark.json"
    payload = {
        "name": "sample-bench",
        "description": "test",
        "version": "1.0.0",
        "examples": [
            {
                "id": "q1",
                "question": "Q1?",
                "ground_truth_answer": "A1",
                "expected_doc_ids": ["d1"],
                "expected_chunk_ids": ["c1"],
                "metadata": {"topic": "math"},
            },
            {
                "id": "q2",
                "question": "Q2?",
                "ground_truth_answer": "A2",
                "expected_doc_ids": ["d2"],
                "expected_chunk_ids": ["c2"],
                "metadata": {"topic": "science"},
            },
        ],
    }
    file_path.write_text(json.dumps(payload))

    dataset = EvaluationDataset.from_json(file_path)
    assert len(dataset) == 2
    assert dataset[0].id == "q1"
    assert dataset[1].question == "Q2?"

    # Filter
    filtered = dataset.filter(lambda ex: ex.metadata.get("topic") == "math")
    assert len(filtered) == 1
    assert filtered[0].id == "q1"

    # Sample
    sampled = dataset.sample(1, seed=123)
    assert len(sampled) == 1

    # Round trip
    out_file = tmp_path / "out.json"
    dataset.to_json(out_file)
    reloaded = EvaluationDataset.from_json(out_file)
    assert len(reloaded) == 2


def test_eval_dataset_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        EvaluationDataset.from_json("nonexistent_path_signalrag.json")


def test_eval_dataset_benchmark_file():
    benchmark_path = Path("eval/questions.json")
    assert benchmark_path.exists()
    dataset = EvaluationDataset.from_json(benchmark_path)
    assert len(dataset) >= 15
    for ex in dataset:
        assert ex.id.startswith("eval-")
        assert len(ex.question) > 10
        assert len(ex.ground_truth_answer) > 10
