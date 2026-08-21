# Interview walkthrough

## Two-minute explanation

This repository demonstrates that RAG testing requires separate validation of retrieval, context, generation, citations, hallucination, performance and regression quality. It uses deterministic metrics wherever possible and model-based evaluation only for semantic dimensions that genuinely require judgment.

The offline path loads versioned Acme policies, chunks them, runs dense plus BM25 retrieval, fuses ranks, optionally reranks, builds bounded context, generates an extractive cited response, and sends every layer into a quality gate. Eighty controlled cases and case-level reports make failures reproducible.

## Five-minute walkthrough

1. Open `data/knowledge/hr/annual-leave.md` and show version/status/region metadata.
2. Open `retrieval.jsonl` and show expected document and chunk labels.
3. Run a query, then explain dense, BM25 and RRF component scores.
4. Open an HTML failure to show expected sources, retrieved sources, answer, citations and reason codes.
5. Run the benchmark to compare dense, hybrid, reranked and top-k configurations on one dataset.
6. Change a threshold and show the regression gate rather than judging prose manually.

## Concise interview answers

- **What is RAG?** Retrieval supplies external evidence to a generator at request time.
- **Why test retrieval separately?** Generation cannot use evidence that retrieval missed, and a fluent answer can conceal that miss.
- **Precision versus recall?** Precision measures relevant share of returned items; recall measures recovered share of all relevant items.
- **Recall@K?** Relevant items present in the first K results divided by all labelled relevant items.
- **MRR?** Mean inverse rank of the first relevant item; it rewards early evidence.
- **Why hybrid search?** Dense handles semantic similarity; BM25 preserves exact terms, codes and numbers.
- **What is reranking?** A more precise scorer reorders a small candidate set before context assembly.
- **How select chunk size?** Compare several sizes on the same dataset for recall, precision, tokens and latency.
- **What does top-k affect?** Recall, distraction, context cost and latency.
- **How test hallucination?** Label required/forbidden claims, map claims to context, test missing evidence and validate citations.
- **Groundedness versus correctness?** Groundedness means supported by context; context itself could still be wrong.
- **Faithfulness versus relevance?** Faithfulness preserves evidence; relevance addresses the question.
- **How evaluate citations?** Existence, correctness, completeness and claim-to-source attribution.
- **Missing evidence?** State what is known and explicitly refuse the unsupported part.
- **Why can good retrieval still hallucinate?** The generator may ignore, misread or add claims beyond retrieved context.
- **How regression-test RAG?** Freeze dataset/config provenance, measure candidate metrics and compare against gates/baseline.
- **Compare embeddings?** Hold corpus, chunks, queries and labels fixed; compare retrieval plus latency/resource cost.
- **Evaluate reranker?** Compare pre/post ranking metrics and add reranker latency.
- **Why not judge everything with an LLM?** Deterministic facts and IDs are cheaper, reproducible and auditable; judges add bias and variance.
- **Document prompt injection?** Treat source text as untrusted data and test that it cannot change instructions or citations.
- **Outdated knowledge?** Version metadata, effective dates, current-status filtering and conflict cases.
- **Cost/latency trade-off?** Report quality with p50/p95/p99, tokens and cost per successful grounded answer.
- **Millions of documents?** Managed/partitioned search, incremental indexing and the same evaluation contract at scale.
- **Azure AI Search/OpenSearch?** Implement the retriever protocol and rerun the unchanged labelled suite.
- **Tenant isolation?** Apply tenant/access filters in retrieval and prove cross-tenant negatives.

