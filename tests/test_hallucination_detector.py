"""Tests for hallucination detection and claim verification."""

from signalrag.generation.hallucination import HallucinationDetector
from signalrag.models.chunk import Chunk


def test_hallucination_detector_pass():
    chunk = Chunk(
        id="c1",
        document_id="d1",
        text="SignalRAG utilizes BM25 lexical search and semantic vector embeddings for hybrid retrieval.",
    )
    answer = "SignalRAG uses BM25 lexical search. It also supports semantic vector embeddings."

    detector = HallucinationDetector(support_threshold=0.4)
    report = detector.verify_answer(answer, [chunk])

    assert report.verdict == "PASS"
    assert report.total_claims == 2
    assert report.supported_claims == 2
    assert report.unsupported_claims == 0
    assert report.faithfulness_ratio == 1.0


def test_hallucination_detector_fail():
    chunk = Chunk(
        id="c1",
        document_id="d1",
        text="SignalRAG is written in Python.",
    )
    # Answer contains fabricated claims not present in chunk
    answer = "SignalRAG was invented by NASA in 1969 to guide the Apollo moon mission."

    detector = HallucinationDetector(support_threshold=0.5)
    report = detector.verify_answer(answer, [chunk])

    assert report.verdict == "FAIL"
    assert report.supported_claims == 0
    assert report.unsupported_claims == 1
    assert report.faithfulness_ratio == 0.0
