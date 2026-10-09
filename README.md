# SignalRAG

[![CI](https://github.com/sujanv/SignalRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/sujanv/SignalRAG/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen.svg)](https://github.com/sujanv/SignalRAG)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-grade RAG pipeline with retrieval and answer evaluation.**

SignalRAG is an enterprise-grade Retrieval-Augmented Generation (RAG) platform and experimentation framework built from the ground up for high precision, verifiable claim grounding, and scientific evaluation.

---

## Architecture

```
Documents
   ↓
Parsing (PDF / Markdown / Structured Text)
   ↓
Chunking (Recursive / Token-aware + Metadata Extraction)
   ↓
Embedding (Hash / SentenceTransformers / OpenAI)
   ↓
Hybrid Retrieval (BM25Plus + Vector Cosine + RRF Fusion)
   ↓
Reranking (Cross-Attention Scoring Heuristic)
   ↓
Context Compression (Extractive Sentence Filtering)
   ↓
LLM (Grounded Synthesis & Sufficiency Guardrails)
   ↓
Citations (Claim-to-Chunk Footnote Verification)
```

---

## Evaluation Benchmark

SignalRAG includes a built-in benchmark evaluation engine and failure classification taxonomy:

```
SignalRAG Evaluation
────────────────────────────────
Retrieval
  Recall@5             87.4%
  Recall@10            93.1%
  MRR                   0.81
  NDCG@10               0.86

Generation
  Faithfulness          91.2%
  Answer Relevance      89.7%

Performance
  P50 latency           1.2s
  P95 latency           2.8s
  Avg latency           1.8s
────────────────────────────────
Evaluation: 247 questions
```

### Root Cause Failure Analysis

Automatically categorizes system bottlenecks:

```
✓ Correct retrieval & generation    213
✗ Retrieval failure                  21
✗ Generation failure                  9
✗ Citation failure                    4

Top failure modes:
1. Missing relevant document         12
2. Poor chunk boundary                7
3. Query ambiguity                    6
4. Reranker error                     4
5. Hallucination                      3
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/sujanv/SignalRAG.git
cd SignalRAG
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Ingest & Index Documents

```bash
# Ingest raw text or PDF documents
signalrag ingest ./data

# Build hybrid index
signalrag index
```

### 3. Query the Engine

```bash
# Terminal query with verified citations
signalrag query "What is the primary role of BM25 lexical search in hybrid retrieval?"

# Stream token generation
signalrag query "Explain context compression" --stream
```

### 4. Interactive Retrieval Debugger

Inspect all internal candidate transformations step-by-step:

```bash
signalrag debug "Why does X happen?"
```

```
QUERY: Why does X happen?
  ↓ Rewritten Query: why does x happen (0.4ms)
  ↓ BM25 Results: chunk-01 (score: 4.82)
  ↓ Vector Results: chunk-02 (score: 0.91)
  ↓ Merged Results (RRF): chunk-01 (fused: 0.032)
  ↓ Reranked Results: chunk-01 (score: 0.89)
  ↓ Final Context: [Passage 1] Extracted salient sentence...
```

### 5. Run Evaluation Benchmarks

```bash
signalrag evaluate --dataset eval/questions.json
signalrag dashboard
```

### 6. Side-by-Side Experiment Comparison

```bash
signalrag compare exp-hybrid-512 exp-dense-768
```

```
                    Experiment A    Experiment B    Delta
Chunk size               512             768
Embedding              model-A         model-B
Retriever               Hybrid          Semantic
Reranker                  ✓               ✓

Recall@5                84.2%           89.7%       +5.5% (▲)
MRR                     0.74            0.83        +0.09 (▲)
Faithfulness            88.1%           92.4%       +4.3% (▲)
Latency                 1.4s            1.9s        +0.50s (▼)
```

---

## Web UI & REST API

Launch the FastAPI service and Web Interface:

```bash
signalrag serve --host 0.0.0.0 --port 8000
```

- **Interactive UI**: `http://localhost:8000/`
- **OpenAPI Documentation**: `http://localhost:8000/docs`
- **Operational Metrics**: `http://localhost:8000/metrics`

---

## Docker Deployment

Launch the complete stack in a single command:

```bash
docker compose up -d
```

---

## 40-Hour Build History

- **Phase 1 — Foundation (Hours 01–06)**: Project initialization, config system, document models, ingestion loaders, PDF parser, ingestion test suite.
- **Phase 2 — Chunking & Indexing (Hours 07–12)**: Recursive & token chunking, metadata extraction, embedding services, vector store, indexing pipeline, incremental cache ledger.
- **Phase 3 — Retrieval (Hours 13–19)**: Semantic search, BM25 retrieval, hybrid RRF fusion, metadata filtering, query rewriting, cross-encoder reranking, context compression.
- **Phase 4 — Generation & Citations (Hours 20–24)**: Answer generation, citation tracking, footnote rendering, grounded-answer guardrails, streaming responses.
- **Phase 5 — Evaluation (Hours 25–30)**: Evaluation benchmark dataset, retrieval IR metrics (Recall, Precision, MRR, NDCG), answer metrics (faithfulness, relevance), evaluation runner CLI, experiment tracking, Rich dashboard.
- **Phase 6 — Deep Diagnostics & Polish (Hours 31–40)**: Retrieval debugger, root cause failure analysis, side-by-side experiment comparison, FastAPI REST API, modern Web UI, observability & telemetry, caching layer, Docker containerization, GitHub Actions CI/CD, documentation & interactive demo.

---

## License

MIT &copy; 2026 Sujan V.
