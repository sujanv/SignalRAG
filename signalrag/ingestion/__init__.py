"""Document ingestion abstractions and loaders."""

from signalrag.ingestion.base import BaseFileLoader, BaseLoader
from signalrag.ingestion.directory_loader import DirectoryLoader
from signalrag.ingestion.text_loader import TextLoader

__all__ = [
    "BaseLoader",
    "BaseFileLoader",
    "TextLoader",
    "DirectoryLoader",
]
