"""Advanced metadata filtering system supporting comparison, membership, and logical operators."""

from collections.abc import Sequence
from typing import Any

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult


class MetadataFilter:
    """Evaluates filtering criteria against chunk metadata.

    Supports:
    - Direct equality: {"author": "Sujan"}
    - Comparison: {"year": {"$gte": 2022}, "page_number": {"$lte": 5}}
    - Membership: {"category": {"$in": ["ml", "ai"]}}
    - Substring: {"source": {"$contains": "research"}}
    - Logical: {"$and": [...], "$or": [...], "$not": {...}}
    """

    def __init__(self, filter_spec: dict[str, Any] | None = None) -> None:
        self.filter_spec = filter_spec or {}

    @classmethod
    def extract_field_value(cls, chunk: Chunk, field_name: str) -> Any:
        """Extract field value from Chunk top-level attributes, metadata attributes, or metadata.extra."""
        # 1. Top-level chunk attributes
        if hasattr(chunk, field_name):
            return getattr(chunk, field_name)

        # 2. ChunkMetadata attributes
        meta = chunk.metadata
        if hasattr(meta, field_name):
            val = getattr(meta, field_name)
            if val is not None:
                return val

        # 3. Check extra dict
        if field_name in meta.extra:
            return meta.extra[field_name]

        # 4. Synthesized helpers e.g. 'year' from modified_at or created_at
        if field_name == "year":
            date_val = meta.extra.get("modified_at") or meta.extra.get("created_at")
            if date_val and hasattr(date_val, "year"):
                return date_val.year
            elif isinstance(date_val, str) and len(date_val) >= 4 and date_val[:4].isdigit():
                return int(date_val[:4])

        return None

    def _eval_condition(self, actual: Any, operator_dict: dict[str, Any]) -> bool:
        """Evaluate operator dictionary on actual value."""
        for op, expected in operator_dict.items():
            if op == "$eq":
                if actual != expected:
                    return False
            elif op == "$ne":
                if actual == expected:
                    return False
            elif op == "$in":
                if not isinstance(expected, Sequence) or actual not in expected:
                    return False
            elif op == "$nin":
                if isinstance(expected, Sequence) and actual in expected:
                    return False
            elif op == "$gt":
                if actual is None or actual <= expected:
                    return False
            elif op == "$gte":
                if actual is None or actual < expected:
                    return False
            elif op == "$lt":
                if actual is None or actual >= expected:
                    return False
            elif op == "$lte":
                if actual is None or actual > expected:
                    return False
            elif op == "$contains":
                if actual is None or str(expected).lower() not in str(actual).lower():
                    return False
            else:
                # Unknown operator fallback
                if actual != expected:
                    return False
        return True

    def matches(self, chunk: Chunk) -> bool:
        """Evaluate filter_spec against chunk."""
        if not self.filter_spec:
            return True
        return self._evaluate_spec(chunk, self.filter_spec)

    def _evaluate_spec(self, chunk: Chunk, spec: dict[str, Any]) -> bool:
        """Recursively evaluate filter specification."""
        for key, criterion in spec.items():
            # Logical operators
            if key == "$and":
                if not isinstance(criterion, list) or not all(
                    self._evaluate_spec(chunk, s) for s in criterion
                ):
                    return False
                continue

            if key == "$or":
                if not isinstance(criterion, list) or not any(
                    self._evaluate_spec(chunk, s) for s in criterion
                ):
                    return False
                continue

            if key == "$not":
                if not isinstance(criterion, dict) or self._evaluate_spec(chunk, criterion):
                    return False
                continue

            actual_val = self.extract_field_value(chunk, key)

            if isinstance(criterion, dict):
                # Operator dict
                if not self._eval_condition(actual_val, criterion):
                    return False
            elif isinstance(criterion, list | set | tuple):
                # Implicit $in
                if actual_val not in criterion:
                    return False
            else:
                # Direct equality
                if actual_val is None:
                    return False
                if str(actual_val) != str(criterion):
                    return False

        return True

    def filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Filter a list of Chunk objects."""
        return [c for c in chunks if self.matches(c)]

    def filter_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Filter a list of SearchResult objects, updating ranks."""
        filtered = [r for r in results if self.matches(r.chunk)]
        for idx, r in enumerate(filtered, start=1):
            r.rank = idx
        return filtered
