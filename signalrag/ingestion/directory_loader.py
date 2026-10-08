"""Directory loader for recursively ingesting files from filesystem."""

from collections.abc import Callable, Iterator
from pathlib import Path

from signalrag.ingestion.base import BaseLoader
from signalrag.ingestion.pdf_loader import PDFLoader
from signalrag.ingestion.text_loader import TextLoader
from signalrag.models.document import Document

LoaderFactory = Callable[[Path], BaseLoader]


class DirectoryLoader(BaseLoader):
    """Loads documents from a filesystem directory matching specified patterns."""

    DEFAULT_LOADERS: dict[str, LoaderFactory] = {
        ".txt": lambda p: TextLoader(p),
        ".md": lambda p: TextLoader(p),
        ".markdown": lambda p: TextLoader(p),
        ".rst": lambda p: TextLoader(p),
        ".json": lambda p: TextLoader(p),
        ".csv": lambda p: TextLoader(p),
        ".log": lambda p: TextLoader(p),
        ".pdf": lambda p: PDFLoader(p),
    }

    def __init__(
        self,
        directory_path: str | Path,
        glob_pattern: str = "**/*",
        recursive: bool = True,
        custom_loaders: dict[str, LoaderFactory] | None = None,
        exclude_patterns: list[str] | None = None,
        silent_errors: bool = False,
    ) -> None:
        self.directory_path = Path(directory_path).resolve()
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")
        if not self.directory_path.is_dir():
            raise NotADirectoryError(f"Expected a directory, got: {self.directory_path}")

        self.glob_pattern = glob_pattern
        self.recursive = recursive
        self.silent_errors = silent_errors
        self.exclude_patterns = exclude_patterns or [
            ".git*",
            ".venv*",
            "__pycache__*",
            "*.pyc",
            ".DS_Store",
        ]

        self.loaders: dict[str, LoaderFactory] = dict(self.DEFAULT_LOADERS)
        if custom_loaders:
            self.loaders.update(custom_loaders)

    @classmethod
    def register_loader(cls, extension: str, factory: LoaderFactory) -> None:
        """Globally register a loader for a file extension (e.g. '.pdf')."""
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        cls.DEFAULT_LOADERS[ext] = factory

    def _should_exclude(self, path: Path) -> bool:
        """Check if path matches any exclusion patterns or hidden directories."""
        for part in path.parts:
            if part.startswith(".") and part != ".":
                return True
        for pattern in self.exclude_patterns:
            if path.match(pattern) or any(part.startswith(pattern.rstrip("*")) for part in path.parts):
                return True
        return False

    def lazy_load(self) -> Iterator[Document]:
        """Iterate through directory files and yield parsed documents."""
        paths = self.directory_path.rglob(self.glob_pattern) if self.recursive else self.directory_path.glob(self.glob_pattern)

        for path in sorted(paths):
            if not path.is_file() or self._should_exclude(path):
                continue

            suffix = path.suffix.lower()
            loader_cls = self.loaders.get(suffix)

            if loader_cls is None:
                # If unsupported file format, skip or ignore
                continue

            try:
                loader = loader_cls(path)
                yield from loader.load()
            except Exception as exc:
                if not self.silent_errors:
                    raise RuntimeError(f"Failed to load file '{path}': {exc}") from exc

    def load(self) -> list[Document]:
        """Load and return all documents found in the directory."""
        return list(self.lazy_load())
