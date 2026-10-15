# SignalRAG

[![CI](https://github.com/sujanv/SignalRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/sujanv/SignalRAG/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://github.com/sujanv/SignalRAG)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise-grade RAG platform featuring hybrid search, GraphRAG entity-relation networks, Self-Reflective RAG (Self-RAG), Hypothetical Document Embeddings (HyDE), hierarchical parent-child retrieval, multi-tenant RBAC, semantic caching, and automated regression benchmarking.**

SignalRAG is a production-grade Retrieval-Augmented Generation (RAG) platform, experimentation framework, and diagnostic engine built from the ground up for high precision, verifiable claim grounding, and scientific evaluation.

---

## Architecture

```
                                  User Query
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
      Semantic Query Cache (Hit)             Semantic Query Router
      (Fast Near-Neighbor Return)                        │
                                                         ▼
                                             Multi-Hop Decomposer
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        ▼                                ▼                                ▼
               BM25 Lexical Search               Dense Vector Search            GraphRAG Entity Network
             (Inverted Index Scores)           (HyDE / Parent-Child Chunks)    (Associative Knowledge Graph)
                        │                                │                                │
                        └────────────────────────────────┼────────────────────────────────┘
                                                         ▼
                                           Adaptive Hybrid Fusion
                                       (Query-Conditioned Alpha Normalization)
                                                         │
                                                         ▼
                                            Multi-Tenant RBAC Guard
                                                         │
                                                         ▼
                                           Semantic Chunk Deduplication
                                                         │
                                                         ▼
                                              Cross-Encoder Reranker
                                                         │
                                                         ▼
                                           Context Sentence Compression
                                                         │
                                                         ▼
                                          Dynamic Token Budget Allocator
                                                         │
                                                         ▼
                                           Self-Reflective RAG Evaluator
                                           (Relevance Grading / Re-query)
                                                         │
                                                         ▼
                                                 LLM Generation
                                                         │
                                                         ▼
                                           Footnote Citation Attribution
                                                         │
                                                         ▼
                                          Hallucination & Claim Verifier
                                                         │
                                                         ▼
                                      Prometheus & OpenTelemetry Telemetry
```

---

## Evaluation Benchmark & Quality Gates

SignalRAG includes a built-in benchmark evaluation engine, synthetic dataset generator, failure taxonomy, and automated regression guards:

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

### Root Cause Failure Taxonomy

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

## Comprehensive CLI Reference

SignalRAG provides a unified command-line tool `signalrag` for end-to-end operation, diagnostics, and security:

| Command | Description | Example Usage |
| :--- | :--- | :--- |
| `query` | Execute question answering with cited footnotes | `signalrag query "What is BM25?"` |
| `ingest` | Load and parse documents from filesystem | `signalrag ingest ./data` |
| `index` | Build vector and lexical search indices | `signalrag index` |
| `batch-index` | Multi-threaded concurrent batch indexing | `signalrag batch-index --workers 4 --count 50` |
| `graph-query` | Associative GraphRAG entity-relation traversal | `signalrag graph-query "How does BM25 connect to SignalRAG?"` |
| `self-rag` | Self-reflective retrieval grading & re-query | `signalrag self-rag "query"` |
| `adaptive-fuse` | Dynamic query-conditioned alpha fusion | `signalrag adaptive-fuse "ERR_502_BAD_GATEWAY"` |
| `hierarchical-search` | Small-to-big child-to-parent chunk expansion | `signalrag hierarchical-search --text "..."` |
| `cache-stats` | Semantic query cache hits & telemetry | `signalrag cache-stats` |
| `hyde` | Hypothetical Document Embeddings generation | `signalrag hyde "Explain vector search"` |
| `multihop` | Decompose compound questions into sub-queries | `signalrag multihop "Compare BM25 and vector search"` |
| `route` | Classify query intent to pick retrieval backend | `signalrag route "ERR_CONN_REFUSED 502"` |
| `expand` | Rocchio Pseudo-Relevance Feedback expansion | `signalrag expand "vector index" -k 3` |
| `versions` | Inspect document revision history & active state | `signalrag versions doc_policy_01` |
| `budget` | Fit candidate passages within LLM token window | `signalrag budget --max-tokens 2048 -q "architecture"` |
| `dedup` | Cluster and prune redundant chunk passages | `signalrag dedup --threshold 0.75` |
| `auth-query` | Execute query under multi-tenant RBAC context | `signalrag auth-query -u alice -t tenant_eng -r reader` |
| `verify-answer`| Verify claim-level entailment against evidence | `signalrag verify-answer -a "..." -c "..."` |
| `generate-eval`| Synthesize evaluation pairs with citations | `signalrag generate-eval -d data/samples -o eval/synth.json` |
| `evaluate` | Run full IR and generation metric evaluation | `signalrag evaluate --dataset eval/questions.json` |
| `regression` | Compare candidate run against golden baseline | `signalrag regression --max-drop 0.05` |
| `debug` | Step-by-step diagnostic trace of candidates | `signalrag debug "Why does X happen?"` |
| `compare` | Side-by-side metric comparison between 2 runs | `signalrag compare exp-hybrid exp-dense` |
| `serve` | Launch FastAPI server and Web UI | `signalrag serve --port 8000` |

---

## Client SDKs (Python & Go)

### Python Client SDK (Sync & Async)

```python
from signalrag import SignalRAGClient, AsyncSignalRAGClient

# Synchronous usage
with SignalRAGClient(base_url="http://localhost:8000") as client:
    client.add_document("SignalRAG incorporates hybrid retrieval and claim verification.")
    client.index()
    resp = client.query("What does SignalRAG incorporate?", top_k=3)
    print(resp.answer)
    for c in resp.citations:
        print(f"[{c.index}] {c.text}")

# Asynchronous usage
async with AsyncSignalRAGClient(base_url="http://localhost:8000") as client:
    resp = await client.query("Explain RAG architecture")
    print(resp.answer)
```

### Golang Client SDK & Native Tools

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

#### Native Go CLI and Batch Indexer Tools
```bash
# Query via standalone Go binary
go run clients/go/cmd/signalrag-cli/main.go query -url http://localhost:8000 -k 3 "What is BM25?"

# High-concurrency batch loader in Go
go run clients/go/cmd/signalrag-batch/main.go -url http://localhost:8000 -workers 8 -count 100
```

---

## Production Observability & Prometheus

SignalRAG exports enterprise Prometheus metrics and OpenTelemetry span context:

- **JSON Summary**: `GET http://localhost:8000/metrics`
- **Prometheus Text Format**: `GET http://localhost:8000/metrics/prometheus`

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

## Complete Build Progression (Hours 01–64)

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
  - **Hour 48**: Architecture and production operations guide.
- **Phase 8 — Advanced Reliability & Security (Hours 49–56)**:
  - **Hour 49**: Pseudo-Relevance Feedback (PRF) and query expansion.
  - **Hour 50**: Document versioning and point-in-time temporal filtering.
  - **Hour 51**: Dynamic context window & token budget allocator.
  - **Hour 52**: Semantic chunk deduplication and clustering.
  - **Hour 53**: Multi-tenant isolation and Role-Based Access Control (RBAC).
  - **Hour 54**: Automated RAG regression testing and quality gatekeeper.
  - **Hour 55**: Standalone Golang CLI tool and async Python client SDK.
  - **Hour 56**: Advanced enterprise features documentation.
- **Phase 9 — Context Graphs & Cognitive RAG (Hours 57–64)**:
  - **Hour 57**: Entity-relationship knowledge graph retrieval (GraphRAG).
  - **Hour 58**: Self-Reflective RAG (Self-RAG) with relevance grading & query reformulation.
  - **Hour 59**: Adaptive hybrid fusion weighting with score normalization.
  - **Hour 60**: Hierarchical parent-child small-to-big document retrieval.
  - **Hour 61**: Semantic similarity query cache with vector cosine proximity & TTL.
  - **Hour 62**: Hypothetical Document Embeddings (HyDE) zero-shot passage synthesis.
  - **Hour 63**: Concurrent multi-threaded batch indexing pipeline and Go loader.
  - **Hour 64**: Comprehensive enterprise operations and GraphRAG guide.

---

## License

MIT &copy; 2026 Sujan V.
