"""Base abstractions for document ingestion."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from signalrag.models.document import Document


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return all documents."""
        pass

    def lazy_load(self) -> Iterator[Document]:
        """Lazily yield documents one by one."""
        yield from self.load()


class BaseFileLoader(BaseLoader):
    """Base class for single-file loaders."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path).resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Expected a file path, got: {self.file_path}")
