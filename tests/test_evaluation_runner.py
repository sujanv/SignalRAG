"""Tests for EvaluationRunner benchmarking."""

from pathlib import Path

from signalrag.evaluation.dataset import EvalExample, EvaluationDataset
from signalrag.evaluation.runner import EvaluationRunner
from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


class MockPipeline:
    def retrieve(self, query: str, top_k: int = 5):
        chunk = Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="BM25 is a ranking algorithm.",
            index=0,
            token_count=10,
        )
        return [SearchResult(chunk=chunk, score=0.95)]


def test_evaluation_runner_retrieval_only(tmp_path: Path):
    pipeline = MockPipeline()
    runner = EvaluationRunner(retrieval_pipeline=pipeline)

    example = EvalExample(
        id="q1",
        question="What is BM25?",
        ground_truth_answer="BM25 is a ranking algorithm.",
        expected_chunk_ids=["chunk-1"],
    )
    dataset = EvaluationDataset(examples=[example])

    report = runner.evaluate(dataset=dataset, evaluate_generation=False)
    assert report.total_examples == 1
    assert report.successful_examples == 1
    assert report.retrieval_metrics.recall_at_5 == 1.0
    assert report.retrieval_metrics.mrr == 1.0
    assert report.latency.count == 1

    out_file = tmp_path / "eval_out.json"
    report.to_json(out_file)
    assert out_file.exists()
