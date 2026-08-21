# Observability

Recommended spans are request, embedding, dense retrieval, sparse retrieval, fusion, reranking, context assembly, generation and evaluation. Attributes include retrieval mode, top-k, embedding and reranker identities, provider/model, prompt version, latency and token usage.

The core facade becomes a no-op when OpenTelemetry is not installed. The optional extra can export OTLP to a local Phoenix service. Traces should avoid API keys, authentication headers, personal data and full sensitive source text. Prefer stable IDs, counts, hashes and approved snippets.

Evaluation reports and runtime telemetry answer different questions: reports prove controlled quality, while traces explain individual production behaviour. Both should share trace/evaluation IDs for diagnosis.

