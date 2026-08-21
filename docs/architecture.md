# Architecture

The system separates build-time and evaluation-time concerns. Ingestion normalises trusted local files, preserves governance metadata and creates stable chunks. Retrieval runs dense and BM25 searches over the same chunks, then fuses ranks. Reranking, context assembly and generation are independent stages so each can be benchmarked or replaced.

Each request records the selected documents, ordered chunks, component scores, context budget decisions, provider identity, prompt version, token data when available, and stage latency. Evaluation consumes this evidence rather than asking only whether the final prose looks plausible.

Boundaries use Pydantic models and small protocols. Hosted model SDKs, sentence-transformers, FAISS, OpenTelemetry and Phoenix are optional adapters. Consequently a pull request can validate the full control flow without network access or paid credentials.

For production, move the in-process index and evaluation registry to durable services, authenticate all endpoints, propagate tenant/access claims into retrieval filters, encrypt stored evidence, and attach immutable dataset/configuration provenance to every report.

