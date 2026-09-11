# Live RAGOps Studio

**Live demo:** https://ragops-studio.vercel.app

## Public execution path

The browser demo is intentionally keyless and transparent. Recruiters can edit the corpus and query, then run:

```text
Corpus → chunking → lexical scoring → reranking → top-k context → extractive answer → evaluation
```

The application calculates query coverage, groundedness, ranking scores, latency and failure attribution from the current run rather than displaying pre-written dashboard metrics.

## Engineering proof

This repository is the deeper RAG/LLM evaluation evidence. Use the live demo for the 60-second walkthrough and the repository to inspect datasets, retrieval/evaluation implementation, regression evidence and limitations.

The live app uses a BM25-lite lexical approach; it does not claim production embedding/vector-search quality. A production stack should benchmark embeddings, hybrid retrieval, reranking and representative RAG evaluation before release.
