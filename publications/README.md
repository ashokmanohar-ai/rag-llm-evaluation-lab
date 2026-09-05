# Technical White Papers

This directory is the publication index for technical white papers associated with the **RAG & LLM Evaluation Lab** and authored by **Ashok Kumar Manohar**.

## 1. RAG Quality Engineering

**RAG Quality Engineering: A Practical Framework for Evaluating Retrieval-Augmented Generation Systems**

- [Read the white paper](../WHITEPAPER.md)
- [Citation metadata](../CITATION.cff)
- Version: 1.0
- Published: September 2026

Focus: end-to-end Quality Engineering for RAG systems across retrieval, chunking, hybrid search, reranking, context sufficiency, groundedness, hallucination, citation integrity, prompt injection, authorization, freshness, performance, cost, observability, regression testing and CI/CD quality gates.

---

## 2. RAG Evaluation Beyond Accuracy

**RAG Evaluation Beyond Accuracy: Retrieval Quality, Groundedness, Citation Integrity and Failure Localization**

- [Read the white paper](RAG_EVALUATION_BEYOND_ACCURACY.md)
- [Citation metadata](CITATION_RAG_EVALUATION_BEYOND_ACCURACY.cff)
- Version: 1.0
- Published: September 2026

Focus: Corpus–Retrieve–Rank–Context–Generate–Cite–Gate evaluation; Precision@K, Recall@K, Hit@K, MRR and NDCG; context precision and recall; dense/sparse/hybrid retrieval; reranking; groundedness and faithfulness; required-fact coverage; abstention; citation existence, support and coverage; freshness; authorization; adversarial retrieval; failure localization; latency, token and cost evidence; repeated-run stability; baseline comparison; observability; and release gates.

> **Measurement companion paper:** this publication goes deeper than the practical RAG framework by defining how to isolate failures across corpus, retrieval, ranking, context, generation and citation layers instead of relying on one aggregate answer score.

---

## Reference Implementation

Both publications are supported by the open-source [RAG & LLM Evaluation Lab](https://github.com/ashokmanohar-ai/rag-llm-evaluation-lab), which demonstrates 24 versioned synthetic documents, 80 evaluation cases, stable chunk IDs, fixed and heading-aware chunking, dense retrieval, BM25, Reciprocal Rank Fusion, optional reranking, current/access-scoped document filtering, bounded context with drop traceability, deterministic fact/groundedness/hallucination/citation checks, structured judges, baselines, quality gates, JSON/HTML/JUnit evidence, FastAPI, CLI, Docker and OpenTelemetry/Phoenix-ready observability.

Related repositories extend these patterns into [LLM evaluation](https://github.com/ashokmanohar-ai/llm-quality-evaluation-harness), [Enterprise AI Quality Engineering](https://github.com/ashokmanohar-ai/enterprise-ai-quality-engineering-platform), [AI observability](https://github.com/ashokmanohar-ai/phoenix-llm-observability) and [Continuous Quality Engineering](https://github.com/ashokmanohar-ai/continuous-quality-engineering).

> These are independent practitioner white papers and are not peer-reviewed academic publications, compliance certifications, security certifications, or statements of production readiness.
