"""SignalRAG: Production-grade RAG pipeline with retrieval and answer evaluation."""

from signalrag.generation.engine import SignalRAGEngine

__version__ = "0.1.0"
__all__ = ["SignalRAGEngine", "__version__"]

from signalrag.client import SignalRAGClient

__all__ = ["__version__", "SignalRAGClient"]
