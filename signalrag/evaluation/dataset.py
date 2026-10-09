"""Dataset schemas and utilities for RAG evaluation."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    """Single evaluation example containing query, ground truth, and target IDs."""

    id: str = Field(description="Unique identifier for evaluation instance")
    question: str = Field(description="User question or search query")
    ground_truth_answer: str = Field(description="Ideal or reference factual response")
    expected_doc_ids: list[str] = Field(
        default_factory=list, description="Target document identifiers"
    )
    expected_chunk_ids: list[str] = Field(
        default_factory=list, description="Target chunk identifiers"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata like domain or difficulty"
    )


class EvaluationDataset(BaseModel):
    """Benchmark dataset containing evaluation examples."""

    name: str = Field(default="signalrag-eval-benchmark", description="Benchmark name")
    description: str = Field(
        default="Evaluation dataset for SignalRAG retrieval and generation",
        description="Dataset summary",
    )
    version: str = Field(default="1.0.0", description="Dataset schema version")
    examples: list[EvalExample] = Field(
        default_factory=list, description="List of evaluation instances"
    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EvalExample:
        return self.examples[idx]

    def __iter__(self):
        return iter(self.examples)

    @classmethod
    def from_json(cls, path: str | Path) -> EvaluationDataset:
        """Load benchmark dataset from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {file_path}")

        raw_data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(raw_data, list):
            examples = [EvalExample(**item) for item in raw_data]
            return cls(examples=examples)
        return cls.model_validate(raw_data)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Serialize benchmark dataset to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(self.model_dump_json(indent=indent), encoding="utf-8")

    def filter(self, predicate: Callable[[EvalExample], bool]) -> EvaluationDataset:
        """Return a filtered subset of the dataset."""
        filtered_examples = [ex for ex in self.examples if predicate(ex)]
        return EvaluationDataset(
            name=f"{self.name}-filtered",
            description=self.description,
            version=self.version,
            examples=filtered_examples,
        )

    def sample(self, n: int, seed: int = 42) -> EvaluationDataset:
        """Sample n examples deterministically."""
        rng = random.Random(seed)
        sampled = rng.sample(self.examples, min(n, len(self.examples)))
        return EvaluationDataset(
            name=f"{self.name}-sample-{len(sampled)}",
            description=self.description,
            version=self.version,
            examples=sampled,
        )
