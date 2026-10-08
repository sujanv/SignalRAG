# SignalRAG

> **Production-grade RAG pipeline with retrieval and answer evaluation.**

SignalRAG is designed from the ground up to turn Retrieval-Augmented Generation from an ad-hoc demo into an experimentally evaluated, inspectable retrieval and generation system.

---

## 🏛️ Architecture

```
Documents
   ↓
Parsing (PDF / Markdown / Text)
   ↓
Chunking & Metadata Extraction
   ↓
Embedding (Vector + Lexical)
   ↓
Indexing (Incremental / Hash-checked)
   ↓
Hybrid Retrieval (BM25 + Semantic)
   ↓
Reranking (Cross-Encoder)
   ↓
Context Compression
   ↓
LLM Generation + Citations
   ↓
Evaluation & Failure Diagnostics
```

---

## 📦 Project Structure

```
signalrag/
├── cli/           # Command-line interface
├── core/          # Configuration, logging, exceptions
├── models/        # Document, Chunk, Metadata, SearchResult, Citation
├── ingestion/     # PDF, Markdown, filesystem loaders
├── chunking/      # Token-aware and recursive splitters
├── embeddings/    # Embeddings interface and implementations
├── indexing/      # Vector stores and incremental indexing
├── retrieval/     # BM25, semantic, hybrid, reranking, query rewriting
├── generation/    # LLM generation, citation tracking, guardrails
└── evaluation/    # Metrics (Recall@K, MRR, NDCG, Faithfulness) and experiments
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configuration
```bash
cp .env.example .env
```

### 3. Run Tests
```bash
pytest
```

---

## 📊 Milestone Roadmap

- [x] **Phase 1 — Foundation (Hours 01–06)**: Configuration, Document Models, Ingestion, PDF Parsing, Tests
- [ ] **Phase 2 — Chunking & Indexing (Hours 07–12)**: Splitters, Metadata, Vector Store, Incremental Indexing
- [ ] **Phase 3 — Retrieval (Hours 13–19)**: Semantic, BM25, Hybrid, Filters, Query Rewriting, Reranking
- [ ] **Phase 4 — Generation & Citations (Hours 20–24)**: Answers, Inline Citations, Guardrails, Streaming
- [ ] **Phase 5 — Evaluation (Hours 25–30)**: Retrieval & Answer Metrics, Runner, Experiments
- [ ] **Phase 6 — Deep Diagnostics & Polish (Hours 31–40)**: Debugger, Failure Analysis, API, Web UI
