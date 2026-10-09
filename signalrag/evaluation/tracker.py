"""Experiment tracking system for logging, storing, and loading evaluation runs."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from signalrag.evaluation.runner import EvaluationReport


class ExperimentRun(BaseModel):
    """Snapshot of a benchmark experiment run including parameters and evaluation results."""

    id: str = Field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:8]}")
    name: str = "default_experiment"
    created_at: float = Field(default_factory=time.time)
    git_commit: str = "HEAD"
    config_params: dict[str, Any] = Field(default_factory=dict)
    report: EvaluationReport
    tags: list[str] = Field(default_factory=list)


class ExperimentTracker:
    """Manages experiment persistence on local disk."""

    def __init__(self, storage_dir: str | Path = "storage/experiments"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        name: str,
        config_params: dict[str, Any],
        report: EvaluationReport,
        git_commit: str = "HEAD",
        tags: list[str] | None = None,
        run_id: str | None = None,
    ) -> ExperimentRun:
        """Create and persist an experiment run to storage."""
        run = ExperimentRun(
            id=run_id or f"exp-{uuid.uuid4().hex[:8]}",
            name=name,
            git_commit=git_commit,
            config_params=config_params,
            report=report,
            tags=tags or [],
        )
        file_path = self.storage_dir / f"{run.id}.json"
        file_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return run

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Load an experiment run by identifier."""
        file_path = self.storage_dir / f"{run_id}.json"
        if not file_path.exists():
            return None
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return ExperimentRun.model_validate(data)

    def list_runs(self) -> list[ExperimentRun]:
        """List all tracked experiment runs ordered by creation timestamp."""
        runs = []
        for p in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                runs.append(ExperimentRun.model_validate(data))
            except Exception:
                continue
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs

    def delete_run(self, run_id: str) -> bool:
        """Delete an experiment record from disk."""
        file_path = self.storage_dir / f"{run_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
