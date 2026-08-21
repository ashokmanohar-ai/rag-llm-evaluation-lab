# RAG design decisions

## Hybrid retrieval

Dense search captures related meaning; BM25 preserves exact terms, IDs and numbers. Neither dominates every query. Reciprocal Rank Fusion combines their ranks without pretending their raw scores share a scale.

## Reranking

Retrieving a wider candidate set can improve recall while wasting context. A reranker reorders 20 candidates before five are sent to generation. The CI reranker is deterministic and lexical; the optional CrossEncoder is the realistic open-source experiment.

## Chunking and metadata

Fixed chunking supplies a controlled baseline. Recursive chunking respects headings and paragraph boundaries. Stable IDs make retrieval labels and citations testable. Version, effective date, status, region and access group make governance part of retrieval rather than a post-processing promise.

## Context budget and refusal

The context builder preserves rank, removes near duplicates, enforces a token proxy budget and records dropped chunks. When evidence is missing, the grounded prompt requires a refusal. Silence, invisible truncation and guessing are treated as defects.

