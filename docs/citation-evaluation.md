# Citation evaluation

Citation existence verifies that an ID maps to a retrieved internal chunk. Correctness verifies that the cited source is the expected source for the claim. Completeness checks whether evidence-requiring claims have citations. Attribution records `claim → citation → source text` for human review.

An existing source can still be a wrong citation. A correct citation on one sentence does not cover another unsupported claim. IDs printed inside a document are untrusted strings; only IDs assigned by ingestion are eligible. The test corpus covers valid, wrong, missing, invented and content-inconsistent citations.

