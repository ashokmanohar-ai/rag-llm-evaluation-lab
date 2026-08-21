# Retrieval evaluation

For the top `k` results:

\[
Precision@k=\frac{|Retrieved_k \cap Relevant|}{k}
\]

\[
Recall@k=\frac{|Retrieved_k \cap Relevant|}{|Relevant|}
\]

Hit@k is one if any relevant item appears. Reciprocal rank is `1/r` for the first relevant rank, and MRR averages it across queries. NDCG discounts graded relevance by rank and normalises against the ideal ordering.

Recall answers whether evidence was missed. Precision estimates distraction and context waste. Higher top-k often increases recall but lowers precision, increases context tokens and adds latency. Results therefore include per-case expected/retrieved IDs, ranks and scores, not only aggregates.

Dense, sparse, hybrid and reranked configurations must use the same corpus and dataset. Failures receive `RELEVANT_DOCUMENT_MISSED` or `IRRELEVANT_CONTEXT` reason codes for deterministic triage.

