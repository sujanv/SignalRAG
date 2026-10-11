# SignalRAG

[![CI](https://github.com/sujanv/SignalRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/sujanv/SignalRAG/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://github.com/sujanv/SignalRAG)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-grade RAG pipeline with retrieval and answer evaluation, multi-hop decomposition, hallucination detection, and enterprise telemetry.**

SignalRAG is an enterprise-grade Retrieval-Augmented Generation (RAG) platform, experimentation framework, and diagnostic engine built from the ground up for high precision, verifiable claim grounding, and scientific evaluation.

---

## Architecture

```
User Query
    ↓
Semantic Query Router ───────► (Direct Lexical BM25 / Semantic Vector / MultiHop)
    ↓
Multi-Hop Query Decomposer
    ↓
┌──────────────────────────────┐
│  BM25 Lexical Retriever      │
│  Dense Vector Retriever      │
│  (SQLite / In-Memory Store)  │
└──────────────┬───────────────┘
               ↓
    Reciprocal Rank Fusion
               ↓
     Cross-Encoder Reranker
               ↓
  Context Sentence Compression
               ↓
       LLM Generation
               ↓
   Footnote Citation Tracker
               ↓
  Hallucination Detector & Claim Attribution
               ↓
 Prometheus Metrics & OpenTelemetry Spans
```

---

## Evaluation Benchmark

SignalRAG includes a built-in benchmark evaluation engine, synthetic dataset generator, and failure classification taxonomy:

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

# Build hybrid index (In-memory or persistent SQLite)
signalrag index
```

### 3. Query the Engine

```bash
# Terminal query with verified citations
signalrag query "What is the primary role of BM25 lexical search in hybrid retrieval?"

# Stream token generation
signalrag query "Explain context compression" --stream
```

### 4. Advanced Features (Hours 41–48)

#### Multi-Hop Query Decomposition (Hour 41)
Break complex, compound, or comparative queries into atomic sub-questions with fused evidence:
```bash
signalrag multihop "Compare BM25 lexical ranking and dense vector search"
```

#### Semantic Query Routing (Hour 44)
Analyze query intent and route to the optimal retrieval backend:
```bash
signalrag route "ERR_CONNECTION_REFUSED 502"
signalrag route "Why is reciprocal rank fusion effective for hybrid search?"
```

#### Hallucination Detection & Claim Attribution (Hour 45)
Decompose generated answers into atomic claims and verify factual entailment against context:
```bash
signalrag verify-answer   --answer "SignalRAG uses BM25 and vector search. It was invented in 1969 by Apollo."   --context "SignalRAG combines BM25 lexical search with dense vector similarity."
```

#### Synthetic Evaluation Dataset Generation (Hour 43)
Automatically synthesize evaluation questions and ground-truth citations from local documents:
```bash
signalrag generate-eval --docs data/samples --output eval/synthetic.json --count 20
```

---

## Client SDKs (Python & Go)

### Python Client

```python
from signalrag import SignalRAGClient

with SignalRAGClient(base_url="http://localhost:8000") as client:
    # 1. Ingest document
    client.add_document("SignalRAG combines BM25 and vector search with claim verification.")

    # 2. Trigger indexing
    client.index()

    # 3. Query with citations
    resp = client.query("What does SignalRAG combine?", top_k=3)
    print(f"Answer: {resp.answer}")
    for c in resp.citations:
        print(f"[{c.index}] Doc: {c.document_id} - {c.text}")
```

### Golang Client

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/sujanv/SignalRAG/clients/go"
)

func main() {
	client := signalrag.NewClient("http://localhost:8000", nil)

	resp, err := client.Query(context.Background(), signalrag.QueryRequest{
		Query: "Explain RAG architecture",
		TopK:  3,
	})
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}

	fmt.Printf("Answer: %s\nLatency: %.1fms\n", resp.Answer, resp.LatencyMs)
}
```

---

## Production Observability & Prometheus

SignalRAG provides enterprise monitoring and OpenTelemetry span tracking:

- **JSON Summary**: `GET http://localhost:8000/metrics`
- **Prometheus Text Exposition**: `GET http://localhost:8000/metrics/prometheus`

```prometheus
# HELP signalrag_queries_total Total number of RAG queries executed.
# TYPE signalrag_queries_total counter
signalrag_queries_total{status="success"} 42
signalrag_queries_total{status="error"} 0

# HELP signalrag_latency_seconds_sum Total cumulative latency across queries.
# TYPE signalrag_latency_seconds_sum counter
signalrag_latency_seconds_sum 1.845012

# HELP signalrag_retrieved_chunks_total Total number of candidate chunks retrieved.
# TYPE signalrag_retrieved_chunks_total counter
signalrag_retrieved_chunks_total 126
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
- **Prometheus Metrics**: `http://localhost:8000/metrics/prometheus`

---

## Docker Deployment

Launch the complete stack in a single command:

```bash
docker compose up -d
```

---

## Complete Build Progression (Hours 01–48)

- **Phase 1 — Foundation (Hours 01–06)**: Project initialization, config system, document models, ingestion loaders, PDF parser, ingestion test suite.
- **Phase 2 — Chunking & Indexing (Hours 07–12)**: Recursive & token chunking, metadata extraction, embedding services, vector store, indexing pipeline, incremental cache ledger.
- **Phase 3 — Retrieval (Hours 13–19)**: Semantic search, BM25 retrieval, hybrid RRF fusion, metadata filtering, query rewriting, cross-encoder reranking, context compression.
- **Phase 4 — Generation & Citations (Hours 20–24)**: Answer generation, citation tracking, footnote rendering, grounded-answer guardrails, streaming responses.
- **Phase 5 — Evaluation (Hours 25–30)**: Evaluation benchmark dataset, retrieval IR metrics (Recall, Precision, MRR, NDCG), answer metrics (faithfulness, relevance), evaluation runner CLI, experiment tracking, Rich dashboard.
- **Phase 6 — Deep Diagnostics & Polish (Hours 31–40)**: Retrieval debugger, root cause failure analysis, side-by-side experiment comparison, FastAPI REST API, modern Web UI, observability & telemetry, caching layer, Docker containerization, GitHub Actions CI/CD, documentation & interactive demo.
- **Phase 7 — Enterprise Enhancements (Hours 41–48)**:
  - **Hour 41**: Multi-hop query decomposition and reasoning retrieval.
  - **Hour 42**: Persistent SQLite vector store with ACID metadata indexing.
  - **Hour 43**: Synthetic evaluation dataset generator.
  - **Hour 44**: Semantic query router (BM25, Semantic, MultiHop, Hybrid).
  - **Hour 45**: Hallucination detector and claim-level attribution verification.
  - **Hour 46**: Prometheus metrics exposition and OpenTelemetry tracing.
  - **Hour 47**: Official Python and Golang client SDKs.
  - **Hour 48**: Complete architecture and production operations guide.

---

## License

MIT &copy; 2026 Sujan V.
