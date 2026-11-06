# SignalRAG

[![CI](https://github.com/sujanv/SignalRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/sujanv/SignalRAG/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](https://github.com/sujanv/SignalRAG)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise-grade RAG platform featuring hybrid search, GraphRAG entity-relation networks, Corrective RAG (CRAG), Contextual Retrieval, Self-Reflective RAG (Self-RAG), Hypothetical Document Embeddings (HyDE), hierarchical parent-child retrieval, Server-Sent Events (SSE) streaming, adversarial red-teaming, multi-tenant RBAC, and automated regression benchmarking.**

SignalRAG is a production-grade Retrieval-Augmented Generation (RAG) platform, experimentation framework, and diagnostic engine built from the ground up for high precision, verifiable claim grounding, and scientific evaluation.

---

## Architecture

```
                                  User Query / Conversational Chat
                                                │
                             ┌──────────────────┴──────────────────┐
                             ▼                                     ▼
                Semantic Query Cache (Hit)             Conversational Memory / Coreference
                (Fast Near-Neighbor Return)                        │
                                                                   ▼
                                                         Semantic Query Router
                                                                   │
                                                       Multi-Hop Decomposer
                                                                   │
                         ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
                         ▼                                         ▼                                         ▼
                BM25 Lexical Search                       Dense Vector Search                     GraphRAG Entity Network
             (Contextual Chunk Headers)               (HyDE / Parent-Child Chunks)             (Associative Knowledge Graph)
                         │                                         │                                         │
                         └─────────────────────────────────────────┼─────────────────────────────────────────┘
                                                                   ▼
                                                     Adaptive Hybrid Fusion
                                                 (Query-Conditioned Alpha Weighting)
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
                                                      Corrective RAG (CRAG) Evaluator
                                                     (High / Ambiguous / Web Fallback)
                                                                   │
                                                                   ▼
                                                     Self-Reflective RAG Evaluator
                                                     (Relevance Grading / Re-query)
                                                                   │
                                                                   ▼
                                                            LLM Generation
                                                                   │
                                 ┌─────────────────────────────────┴─────────────────────────────────┐
                                 ▼                                                                   ▼
                    Footnote Citation Attribution                                       Server-Sent Events (SSE) Stream
                                 │                                                       (Real-Time Token & Source Delivery)
                                 ▼
                     Hallucination & Claim Verifier
                                 │
                                 ▼
                     Adversarial Red-Team Defense
                                 │
                                 ▼
                 Prometheus & OpenTelemetry Telemetry
```

---

## Evaluation Benchmark & Quality Gates

SignalRAG includes a built-in benchmark evaluation engine, synthetic dataset generator, failure taxonomy, automated regression guards, and adversarial red-teaming:

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

Safety & Robustness
  Adversarial Defense   100.0%
  Hallucination Refusal 96.8%

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
| `stream` | Stream tokens in real-time via Server-Sent Events | `signalrag stream "Explain vector search"` |
| `chat` | Conversational multi-turn session with coreference | `signalrag chat "Can it handle circular dependencies?" -s user1` |
| `crag` | Corrective RAG retrieval confidence evaluation & web fallback | `signalrag crag "quantum computing principles"` |
| `contextualize` | Enrich document chunks with situational context headers | `signalrag contextualize --id doc1 --text "..."` |
| `cluster` | Unsupervised corpus topic clustering & centroid discovery | `signalrag cluster --k 3` |
| `explain` | Token-level saliency attribution & ranking explanation | `signalrag explain "hybrid search" -t "..."` |
| `redteam` | Automated adversarial prompt injection & refusal audit | `signalrag redteam` |
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

#### Native Go Streaming, Batch Indexer, and Query CLI Tools
```bash
# Real-time token streaming client in Go
go run clients/go/cmd/signalrag-stream/main.go -url http://localhost:8000/query/stream -q "Explain streaming retrieval"

# Standalone query tool in Go
go run clients/go/cmd/signalrag-cli/main.go query -url http://localhost:8000 -k 3 "What is BM25?"

# High-concurrency batch loader in Go
go run clients/go/cmd/signalrag-batch/main.go -url http://localhost:8000 -workers 8 -count 100
```

---

## Complete Build Progression (Hours 01–72)

- **Phase 1 — Foundation (Hours 01–06)**: Project initialization, config system, document models, ingestion loaders, PDF parser, ingestion test suite.
- **Phase 2 — Chunking & Indexing (Hours 07–12)**: Recursive & token chunking, metadata extraction, embedding services, vector store, indexing pipeline, incremental cache ledger.
- **Phase 3 — Retrieval (Hours 13–19)**: Semantic search, BM25 retrieval, hybrid RRF fusion, metadata filtering, query rewriting, cross-encoder reranking, context compression.
- **Phase 4 — Generation & Citations (Hours 20–24)**: Answer generation, citation tracking, footnote rendering, grounded-answer guardrails, streaming responses.
- **Phase 5 — Evaluation (Hours 25–30)**: Evaluation benchmark dataset, retrieval IR metrics (Recall, Precision, MRR, NDCG), answer metrics (faithfulness, relevance), evaluation runner CLI, experiment tracking, Rich dashboard.
- **Phase 6 — Deep Diagnostics & Polish (Hours 31–40)**: Retrieval debugger, root cause failure analysis, side-by-side experiment comparison, FastAPI REST API, modern Web UI, observability & telemetry, caching layer, Docker containerization, GitHub Actions CI/CD, documentation & interactive demo.
- **Phase 7 — Enterprise Enhancements (Hours 41–48)**: Multi-hop query decomposition, persistent SQLite vector store, synthetic evaluation dataset generator, semantic query router, hallucination detection & claim attribution, Prometheus metrics & OpenTelemetry tracing, Python & Go client SDKs, architecture and operations guide.
- **Phase 8 — Advanced Reliability & Security (Hours 49–56)**: Pseudo-Relevance Feedback (PRF), document versioning & temporal point-in-time filtering, dynamic token budget allocator, semantic chunk deduplication, multi-tenant RBAC security, automated regression benchmarking, standalone Go CLI, enterprise documentation.
- **Phase 9 — Context Graphs & Cognitive RAG (Hours 57–64)**: GraphRAG entity-relation associative retrieval, Self-Reflective RAG (Self-RAG), adaptive hybrid fusion weighting, hierarchical parent-child retrieval, semantic similarity query cache, HyDE generator, concurrent multi-threaded batch indexing, GraphRAG architecture documentation.
- **Phase 10 — Autonomous Agentic Workflows & Enterprise Scale (Hours 65–72)**:
  - **Hour 65**: Corrective RAG (CRAG) retrieval evaluation and fallback web synthesis.
  - **Hour 66**: Contextual BM25 and dense chunk header enrichment.
  - **Hour 67**: Agentic conversational memory and multi-turn state.
  - **Hour 68**: Semantic document clustering and topic modeling.
  - **Hour 69**: Explainable retrieval attribution and feature heatmaps.
  - **Hour 70**: Streaming SSE event source API and native Go subscriber.
  - **Hour 71**: Autonomous evaluation red-teaming adversarial generator.
  - **Hour 72**: CRAG agentic workflows and streaming architecture documentation.

---

## License

MIT &copy; 2026 Sujan V.
