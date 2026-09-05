# RAG Evaluation Beyond Accuracy

## Retrieval Quality, Groundedness, Citation Integrity and Failure Localization

**Technical White Paper — Version 1.0**  
**September 2026**

**Author:** Ashok Kumar Manohar  
**GitHub:** [ashokmanohar-ai](https://github.com/ashokmanohar-ai)  
**Primary reference implementation:** [RAG & LLM Evaluation Lab](https://github.com/ashokmanohar-ai/rag-llm-evaluation-lab)  
**Related implementations:** [Enterprise AI Quality Engineering Platform](https://github.com/ashokmanohar-ai/enterprise-ai-quality-engineering-platform), [LLM Quality Evaluation Harness](https://github.com/ashokmanohar-ai/llm-quality-evaluation-harness), [Phoenix LLM Observability](https://github.com/ashokmanohar-ai/phoenix-llm-observability), and [Continuous Quality Engineering](https://github.com/ashokmanohar-ai/continuous-quality-engineering)

> **Publication note:** This is an independent practitioner white paper supported by open-source reference implementations. It is not a peer-reviewed academic publication, legal opinion, compliance certification, security certification, or statement of production readiness. Retrieval metrics, evaluation datasets, thresholds, judges, access-control policies, freshness rules and production gates must be calibrated for each organization, corpus and use case.

---

## Abstract

Retrieval-Augmented Generation (RAG) systems are often evaluated as if they were ordinary question-answering models: provide a question, inspect the final answer, and assign a correctness score. That approach hides the main engineering value of RAG—the ability to trace an answer back through retrieval, ranking, context construction and source evidence—and it makes failures difficult to diagnose.

A RAG answer can be wrong because the correct document was never indexed, because chunking split a required fact, because dense retrieval missed a lexical match, because sparse retrieval over-weighted noisy terms, because fusion or reranking placed the wrong passage first, because a newer document was filtered out, because an unauthorized document was retrieved, because the context builder dropped a critical chunk, because the generator ignored good evidence, or because citations pointed to sources that did not support the claim. A single answer-accuracy score cannot distinguish those failure classes.

This white paper presents **RAG Evaluation Beyond Accuracy** as an evidence-driven Quality Engineering discipline. It proposes a **Corpus–Retrieve–Rank–Context–Generate–Cite–Gate model** that evaluates every important boundary in a RAG system and preserves enough evidence to localize failures.

The framework separates **retrieval quality**, **context quality**, **generation quality**, **citation integrity**, **security and authorization**, and **operational quality**. It combines deterministic information-retrieval metrics such as Precision@K, Recall@K, Hit@K, MRR and NDCG with context-level measures, claim-level groundedness, citation coverage, source validity, freshness, access scope, latency, token use, cost, repeated-run stability and baseline regression.

The companion open-source implementation demonstrates versioned documents, stable chunk IDs, fixed and heading-aware chunking, dense retrieval, BM25, Reciprocal Rank Fusion, optional reranking, current-document filtering, bounded context, deterministic groundedness and citation checks, 80 evaluation cases, baselines, quality gates, JSON/HTML/JUnit evidence, FastAPI, CLI, Docker and OpenTelemetry/Phoenix-ready observability.

The central proposition is:

> **A RAG system is trustworthy only when the team can prove not merely that the answer was acceptable, but that the right evidence was available, retrieved, ranked, authorized, current, preserved in context, used correctly and cited truthfully.**

---

## 1. Executive Summary

RAG quality is a chain of evidence:

```text
Authoritative Corpus
        ↓
Ingestion + Metadata
        ↓
Chunking + Embeddings / Sparse Index
        ↓
Candidate Retrieval
        ↓
Fusion / Reranking
        ↓
Context Selection
        ↓
Generation / Abstention
        ↓
Citations
        ↓
Quality Gate
```

A defect at any stage can produce a bad answer. The first goal of RAG evaluation is therefore not to calculate one universal score. It is to answer:

1. **Corpus:** Was the required source present, current and permitted?
2. **Retrieval:** Was relevant evidence found?
3. **Ranking:** Was the best evidence placed high enough?
4. **Context:** Did the generator actually receive sufficient, non-redundant evidence?
5. **Generation:** Were the claims correct and supported?
6. **Citation:** Did each important claim point to a real source that supports it?
7. **Operations:** Did the pipeline meet latency, token and cost objectives?
8. **Governance:** Is the evidence complete enough for release?

The quality strategy should localize failure before attempting remediation.

---

## 2. Why Final-Answer Accuracy Is Not Enough

Suppose two systems both answer 85% of a test set correctly.

System A retrieves the correct source almost every time but occasionally produces an unsupported claim. System B often retrieves the wrong source but the model answers from memorized knowledge. The same answer-accuracy score hides two very different risks.

A production team needs to know **which component failed** because the remediation differs:

- improve ingestion;
- change chunking;
- tune retrieval;
- add hybrid search;
- adjust fusion;
- add a reranker;
- change context budget;
- improve the prompt;
- strengthen abstention;
- fix citation generation;
- repair metadata/access filtering;
- update the evaluation dataset.

---

## 3. The Corpus–Retrieve–Rank–Context–Generate–Cite–Gate Model

The proposed model evaluates seven linked control points:

| Stage | Primary question | Typical evidence |
|---|---|---|
| Corpus | Is the right source available and allowed? | document ID, version, status, region, access group |
| Retrieve | Did the retriever find relevant evidence? | retrieved IDs, scores, top-K metrics |
| Rank | Did it order evidence usefully? | rank positions, MRR, NDCG, reranker output |
| Context | Did the generator receive sufficient evidence? | selected chunks, drops, token budget, redundancy |
| Generate | Are claims correct and grounded? | required facts, unsupported claims, abstention |
| Cite | Are citations real and supportive? | source existence, claim-source support, coverage |
| Gate | Is the evidence sufficient for release? | thresholds, baseline deltas, missing evidence |

---

## 4. Evaluation Starts with a Versioned Corpus

A RAG evaluation cannot be reproducible if the knowledge base changes invisibly.

At minimum, retain:

- stable document IDs;
- document version;
- publication/effective date;
- status such as current, superseded or draft;
- region or jurisdiction;
- department or domain;
- access group / tenant / security label;
- checksum or source revision;
- ingestion timestamp;
- chunking configuration version.

The evaluation report should identify the corpus version used for every run.

---

## 5. Ingestion Quality Is Part of RAG Quality

Before retrieval metrics, test ingestion itself:

- required documents were loaded;
- unsupported files failed explicitly;
- parsing retained meaningful headings, lists and tables;
- metadata survived ingestion;
- superseded or deleted content is handled intentionally;
- duplicate sources are detected;
- unsafe paths or arbitrary remote fetches are rejected;
- the index can be rebuilt reproducibly.

A missing source is not a retriever failure.

---

## 6. Chunking Must Be Evaluated, Not Assumed

Chunk size and boundaries directly affect retrieval and generation.

Test configurations for:

- fixed-size chunks;
- heading-aware chunks;
- recursive splitting;
- overlap;
- table-aware handling;
- list preservation;
- metadata attachment.

Useful evaluation questions include:

- Was a required fact split across chunks?
- Did large chunks introduce unrelated noise?
- Did overlap create excessive duplicates?
- Were headings retained as useful context?
- Did chunking change retrieval ranking?

---

## 7. Retrieval Recall Answers: Did We Miss Necessary Evidence?

For a query with known relevant sources:

\[
Recall@K = \frac{|Relevant \cap Retrieved@K|}{|Relevant|}
\]

Low recall means the pipeline did not retrieve enough required evidence.

Recall-oriented tests are critical for:

- multi-hop questions;
- policy exceptions;
- questions requiring two or more documents;
- region-specific rules;
- temporal/version-sensitive answers.

---

## 8. Retrieval Precision Answers: Did We Retrieve Too Much Noise?

\[
Precision@K = \frac{|Relevant \cap Retrieved@K|}{K}
\]

A system can have high recall but still waste context on irrelevant chunks. Excess noise can increase cost, push useful evidence out of the context window and distract the generator.

Precision and recall must be interpreted together.

---

## 9. Hit@K Is Useful but Incomplete

Hit@K asks whether at least one relevant item appears in the first K results.

It is useful for simple single-source questions, but insufficient when:

- several sources are needed;
- order matters;
- a retrieved source is outdated;
- the source is unauthorized;
- multiple facts must be composed.

---

## 10. MRR Measures the Rank of the First Useful Result

For query set Q:

\[
MRR = \frac{1}{|Q|}\sum_{q \in Q}\frac{1}{rank_q}
\]

MRR is useful when the first correct result strongly influences context construction or user-facing retrieval.

---

## 11. NDCG Captures Graded Ranking Quality

NDCG is useful when relevance is not binary. One chunk may directly answer the question, another may provide supporting background, and a third may be weakly related.

Use graded relevance only when annotation quality is sufficient to justify the extra complexity.

---

## 12. Evaluate Dense and Sparse Retrieval Separately

A hybrid system should not be tested only after fusion.

Measure:

- dense retrieval alone;
- BM25/sparse retrieval alone;
- hybrid fusion;
- hybrid + reranker.

This helps answer whether a regression originates in embeddings, lexical retrieval, fusion or reranking.

---

## 13. Hybrid Search Should Produce Measured Benefit

Hybrid retrieval can combine semantic and lexical strengths, but additional complexity is justified only when it improves representative queries.

Compare candidate configurations on the same dataset and corpus. Avoid changing embeddings, chunking, K, fusion and reranking at the same time when the objective is causal diagnosis.

---

## 14. Reciprocal Rank Fusion Provides Explainable Rank Combination

A common fusion function is:

\[
RRF(d)=\sum_{r \in R}\frac{1}{k+rank_r(d)}
\]

RRF combines rank positions rather than incomparable raw scores. Evaluation should retain the component rankings so a fused result can be explained.

---

## 15. Reranking Needs Its Own Evaluation

A reranker can improve top-K relevance or remove the only useful passage.

Measure:

- candidate recall before reranking;
- top-K precision after reranking;
- rank movement of required sources;
- latency added by reranking;
- failure cases where the reranker suppresses required evidence.

---

## 16. Context Precision Is Not the Same as Retrieval Precision

Retrieval may return 20 candidates while only five are placed into the model context. Context precision evaluates whether the **actual supplied context** is useful and well ordered.

This distinction matters when context builders deduplicate, filter and truncate retrieval results.

---

## 17. Context Recall Measures Sufficiency

Context recall asks whether the supplied context contains enough evidence to support the expected answer.

A retriever can technically find the right document but the context builder may drop it due to token budget or deduplication. That should be localized as a context-selection failure, not retrieval failure.

---

## 18. Context Budget Must Be Observable

Record:

- candidate chunks;
- selected chunks;
- dropped chunks;
- drop reason;
- estimated tokens;
- configured budget;
- deduplication decisions.

Without this evidence, a team cannot explain why a known retrieved source never reached generation.

---

## 19. Redundancy Is a Quality and Cost Problem

Duplicate or near-duplicate chunks consume context without adding information.

Useful measures include:

- duplicate ratio;
- unique-source ratio;
- token utilization;
- relevant-information-per-token proxy.

---

## 20. Groundedness Must Operate at Claim Level

A response can be partially grounded.

Split the response into factual claims and map each claim to supporting evidence.

A practical deterministic representation is:

```json
{
  "claim": "Temporary passwords expire after 24 hours.",
  "supported": true,
  "supporting_chunks": ["IT-PASSWORDS-v3#chunk-04"]
}
```

Unsupported claims should remain visible even when overall answer quality is high.

---

## 21. Faithfulness Measures Consistency with Retrieved Context

Faithfulness asks whether response claims can be supported by the supplied context.

It is different from factual correctness in the external world. A system may faithfully repeat incorrect or outdated context, which is why corpus quality, freshness and authorization remain separate dimensions.

---

## 22. Correctness and Groundedness Are Distinct

Four outcomes are possible:

| Correct? | Grounded? | Interpretation |
|---|---|---|
| Yes | Yes | desired |
| Yes | No | answer may come from model memory; citation risk |
| No | Yes | source/context may be wrong or misunderstood |
| No | No | major generation/retrieval failure |

Release policy should not collapse these outcomes.

---

## 23. Required-Fact Coverage Provides a Deterministic Oracle

For important domain questions, define required facts explicitly.

Example:

```json
{
  "required_facts": {
    "annual_leave_days": 25,
    "carryover_limit": 5
  }
}
```

Normalized fact checks are often more stable and explainable than asking another LLM to assign a generic score.

---

## 24. Abstention Is a First-Class Quality Behavior

When evidence is insufficient, the correct behavior may be:

- "I don't have enough information";
- request clarification;
- route to a human;
- cite available evidence and state uncertainty.

Evaluate **false confidence** and **appropriate abstention** separately.

---

## 25. Unknown Should Not Be Penalized When Evidence Is Missing

A RAG system that refuses to invent an answer can be safer than one that achieves superficially higher answer coverage by hallucinating.

Quality policy should reward safe uncertainty when the corpus genuinely lacks the answer.

---

## 26. Citation Existence Is the Minimum Check

Every emitted citation should map to a real indexed source/chunk.

Reject:

- fabricated IDs;
- deleted documents;
- nonexistent URLs;
- malformed source references.

Citation existence alone does not prove support.

---

## 27. Citation Entailment Checks Whether the Source Supports the Claim

A valid source ID can still be the wrong evidence.

Evaluate whether the cited passage actually supports the associated claim. This may use deterministic fact matching where possible and calibrated semantic evaluation where necessary.

---

## 28. Citation Coverage Measures How Much of the Answer Is Traceable

Important factual claims should have supporting citations.

A practical metric is:

\[
CitationCoverage = \frac{Supported\;cited\;claims}{Citable\;factual\;claims}
\]

Not every conversational sentence needs a citation; the policy should define what is citable.

---

## 29. Citation Precision Penalizes Decorative Citations

A system should not attach many loosely related sources to create an appearance of evidence.

Citation precision asks what fraction of cited sources actually support the associated claims.

---

## 30. Freshness Is a Separate Evaluation Dimension

A source can be relevant and grounded but outdated.

Test:

- current vs superseded policy;
- effective dates;
- version precedence;
- stale caches;
- delayed index refresh;
- conflicting versions.

A grounded answer from superseded evidence is still a quality failure.

---

## 31. Authorization Must Be Enforced Before Generation

RAG access control is not a prompt instruction.

Metadata filters, tenant boundaries and application authorization should constrain retrieval before sensitive content enters model context.

Test cases should include:

- cross-tenant queries;
- region restrictions;
- department restrictions;
- privileged vs ordinary user access;
- missing identity;
- stale or invalid access context.

---

## 32. Retrieval Authorization Is Part of Quality Evidence

A response can be factually correct yet unacceptable if it relied on evidence the caller was not permitted to access.

Evaluation reports should retain the effective identity/scope and the access metadata of selected sources, without exposing secrets.

---

## 33. Prompt Injection in Retrieved Content Must Be Treated as Data

Documents may contain instructions such as "ignore previous instructions" or fabricated citation requests.

The system should treat retrieved content as untrusted evidence, not privileged instructions.

Evaluate:

- indirect prompt injection;
- citation injection;
- policy override attempts;
- hidden instructions;
- malicious HTML/Markdown content.

---

## 34. Data Poisoning and Source Integrity Need Test Coverage

RAG security extends beyond prompt injection.

Test scenarios where:

- a malicious source is inserted;
- source metadata is forged;
- a document is unexpectedly replaced;
- a lower-authority source conflicts with an approved policy;
- duplicate documents create ranking manipulation.

---

## 35. Conflicting Context Requires Explicit Policy

When two sources disagree, the system should not arbitrarily choose one.

Possible policy:

1. prefer current approved sources;
2. prefer higher-authority source type;
3. state the conflict;
4. abstain or escalate when unresolved.

Evaluation should assert the intended conflict-resolution behavior.

---

## 36. Multi-Hop RAG Needs Evidence-Chain Evaluation

Some questions require several retrieval/reasoning steps.

Evaluate:

- whether each required hop retrieved useful evidence;
- whether sub-query reformulation improved recall;
- whether earlier evidence was preserved;
- whether the final answer correctly composed facts across sources;
- hop count and latency;
- safe stop when evidence remains insufficient.

---

## 37. Failure Localization Should Be a First-Class Output

A failed RAG case should produce a likely failure stage.

Example taxonomy:

- `CORPUS_MISSING_SOURCE`
- `INGESTION_PARSE_FAILURE`
- `CHUNK_BOUNDARY_FAILURE`
- `RETRIEVAL_RECALL_FAILURE`
- `RANKING_FAILURE`
- `RERANKER_REGRESSION`
- `CONTEXT_BUDGET_DROP`
- `CONTEXT_REDUNDANCY`
- `STALE_SOURCE`
- `UNAUTHORIZED_SOURCE`
- `GENERATION_UNSUPPORTED_CLAIM`
- `GENERATION_MISSING_FACT`
- `CITATION_MISSING`
- `CITATION_INVALID`
- `CITATION_UNSUPPORTED`
- `ABSTENTION_FAILURE`
- `LATENCY_REGRESSION`
- `COST_REGRESSION`
- `EVALUATION_INFRASTRUCTURE_FAILURE`

---

## 38. Use a Failure Matrix for Diagnosis

| Retrieval | Context | Generation | Citation | Likely area |
|---|---|---|---|---|
| Fail | Fail | Fail | Fail | retriever/corpus |
| Pass | Fail | Fail | Fail | context selection |
| Pass | Pass | Fail | Fail/Pass | generation |
| Pass | Pass | Pass | Fail | citation layer |
| Pass | Pass | Pass | Pass | healthy case |

This simple separation prevents unnecessary model tuning when the real defect is retrieval.

---

## 39. LLM-as-a-Judge Should Be Reserved for Semantic Questions

Use deterministic checks for:

- source existence;
- expected IDs;
- required facts;
- citation validity;
- version/authorization filters;
- latency/tokens;
- schema and metadata.

Use calibrated judges for dimensions such as:

- semantic relevance;
- nuanced completeness;
- answer clarity;
- claim entailment when deterministic matching is insufficient.

Judges should not override deterministic failures.

---

## 40. Judge Calibration Is Required

Before a model judge influences release:

- create human-reviewed examples;
- test agreement;
- measure false-pass and false-fail behavior;
- pin model and prompt versions;
- test position/style bias;
- repeat samples to measure stability;
- define fallback when the judge fails.

---

## 41. Evaluation Datasets Are Product Assets

A strong RAG dataset should include:

- simple fact queries;
- multi-document questions;
- multi-hop questions;
- negative/no-answer cases;
- conflicting sources;
- stale/superseded versions;
- region/tenant/access cases;
- citation-specific cases;
- adversarial injection cases;
- edge terminology and acronyms;
- production-derived sanitized regressions.

Each case needs stable ID, expected evidence and provenance.

---

## 42. Retrieval and Generation Datasets May Need Different Ground Truth

Retrieval cases may specify expected documents/chunks, while QA cases may specify required facts, forbidden claims and acceptable sources.

Do not force every evaluation layer into one overloaded schema.

---

## 43. Baselines Must Be Comparable

Candidate-vs-baseline comparison should hold constant as many variables as possible:

- dataset;
- corpus version;
- environment;
- model/provider where relevant;
- workload;
- judge configuration.

Reject meaningless comparisons between different datasets or materially different corpora unless the change is explicitly the subject of the experiment.

---

## 44. Aggregate Improvement Must Not Hide Critical Regressions

Example:

- Recall@5 improves from 0.82 to 0.86;
- average groundedness improves;
- one security-sensitive query now retrieves another tenant's document.

The release must still fail.

Hard authorization, citation-integrity and critical-correctness gates should remain non-negotiable.

---

## 45. Risk-Based Quality Gates

Example policy categories:

**Hard gates**
- zero unauthorized retrieval;
- zero fabricated citations in critical cases;
- zero unsupported high-impact claims;
- required corpus/index evidence present.

**Threshold gates**
- minimum Recall@K;
- minimum MRR/NDCG;
- minimum required-fact coverage;
- minimum groundedness/citation coverage.

**Warning gates**
- moderate latency increase;
- moderate token/cost increase;
- non-critical rank movement.

---

## 46. Missing Evidence Is Not a Pass

If a mandatory retrieval report, citation evaluator or security dataset did not run, the gate should record missing evidence explicitly.

Do not convert "not measured" into zero defects.

---

## 47. Performance Must Be Measured End to End and by Stage

Useful stage timings include:

- query preprocessing;
- embedding;
- dense retrieval;
- sparse retrieval;
- fusion;
- reranking;
- context construction;
- generation;
- evaluation.

End-to-end latency alone cannot identify the bottleneck.

---

## 48. RAG Cost Is More Than Model Tokens

Relevant cost drivers can include:

- embedding ingestion;
- index storage;
- retrieval service;
- reranker inference;
- generator input/output tokens;
- evaluator/judge calls;
- tracing/storage;
- repeated retries or multi-hop queries.

Report cost only from real usage/pricing evidence. Unknown is better than fabricated precision.

---

## 49. Evaluate Stability Across Repeated Runs

Probabilistic generation can vary even when retrieval is stable.

Repeat representative cases and track:

- answer fact consistency;
- citation consistency;
- abstention consistency;
- judge variance;
- latency variance;
- token variance.

Separate retrieval nondeterminism from generation nondeterminism where possible.

---

## 50. Observability Should Mirror the RAG Pipeline

Recommended spans:

```text
rag.request
  ├─ query.preprocess
  ├─ embedding.query
  ├─ retrieval.dense
  ├─ retrieval.sparse
  ├─ retrieval.fusion
  ├─ rerank
  ├─ context.build
  ├─ generation
  ├─ citation.validate
  └─ evaluation
```

Attach non-sensitive configuration metadata so failures can be reproduced.

---

## 51. Production Failures Should Become Permanent Evaluation Cases

A mature feedback loop is:

```text
Production Failure
      ↓
Locate Trace + Evidence
      ↓
Sanitize Reproduction
      ↓
Classify Failure Stage
      ↓
Add Dataset Case
      ↓
Fix Retrieval / Context / Generation / Citation
      ↓
Compare Against Baseline
      ↓
Retain Regression Permanently
```

---

## 52. Evaluate Configuration Changes as Experiments

Changes worth controlled experiments include:

- chunk size/overlap;
- embedding model;
- sparse tokenizer/analyzer;
- top-K;
- RRF constant;
- reranker;
- metadata policy;
- context budget;
- prompt;
- generator model;
- citation format.

Record the configuration fingerprint with each result.

---

## 53. End-to-End Benchmarks Complement Component Metrics

Component metrics localize failure. End-to-end benchmarks show whether the composed system accomplishes the actual task within an acceptable performance envelope.

Both are required.

MLCommons' 2026 end-to-end RAG benchmark reinforces this architecture by measuring an ingestion workload and an iterative multi-hop Q&A workload across embedding, retrieval, reranking, document grading, sufficiency checking and generation rather than benchmarking only one model.

---

## 54. Enterprise Operating Model

Clear ownership reduces ambiguous failures:

| Area | Primary owner | QE responsibility |
|---|---|---|
| Corpus/source policy | Product/data/domain | verify provenance/freshness cases |
| Ingestion/chunking | RAG/data engineering | benchmark configurations |
| Retrieval/ranking | Search/RAG engineering | retrieval metrics/regressions |
| Generation | AI engineering | groundedness/fact evaluation |
| Authorization | Application/security | cross-scope negative tests |
| Citation | AI/application | source validity/support tests |
| Observability | Platform/SRE | trace completeness/reproducibility |
| Release gates | QE + product + risk | thresholds, blockers, exceptions |

---

## 55. Recommended KPI Set

A balanced production scorecard may include:

**Retrieval**
- Recall@K
- Precision@K
- MRR / NDCG
- required-source hit rate

**Context**
- context recall
- context precision
- duplicate ratio
- token-budget drop rate

**Generation**
- required-fact coverage
- unsupported-claim rate
- abstention correctness
- groundedness / faithfulness

**Citation**
- citation existence
- citation entailment/support
- citation coverage
- fabricated-citation rate

**Security / governance**
- unauthorized-source rate
- stale-source rate
- missing-evidence rate

**Operations**
- p95 end-to-end latency
- p95 retrieval/rerank/generation latency
- tokens per successful answer
- cost per successful answer

---

## 56. Anti-Patterns

Avoid:

1. evaluating only final-answer fluency;
2. using one aggregate RAG score as the release decision;
3. tuning the model when retrieval is the defect;
4. treating a valid citation ID as proof of support;
5. ignoring document version and freshness;
6. enforcing authorization only in the prompt;
7. treating retrieved instructions as trusted system instructions;
8. replacing a baseline merely to make a candidate pass;
9. inventing cost or token values when telemetry is absent;
10. allowing a model judge to override deterministic evidence.

---

## 57. Reference Implementation Mapping

The companion `rag-llm-evaluation-lab` demonstrates:

- 24 versioned synthetic enterprise documents;
- 80 evaluation cases across retrieval, QA, hallucination, citation and adversarial categories;
- fixed and heading-aware chunking;
- dense retrieval and BM25;
- Reciprocal Rank Fusion;
- optional reranking;
- metadata filtering for current/access-scoped documents;
- bounded context with drop traceability;
- deterministic fact/groundedness/hallucination/citation checks;
- structured semantic judge contract;
- baselines and regression gates;
- JSON, HTML and JUnit evidence;
- FastAPI and CLI surfaces;
- Docker and credential-free CI;
- OpenTelemetry/Phoenix-ready instrumentation.

The implementation is a reference system, not proof of universal RAG quality.

---

## 58. Limitations

No RAG evaluation framework can prove correctness for every open-domain question. Ground truth can be incomplete, relevance is sometimes subjective, semantic judges can be biased, corpus policy can change, and real production traffic may differ from curated evaluation data.

The objective is therefore not a perfect universal score. It is a defensible, layered evidence system that makes important failure modes observable and reproducible.

---

## 59. Conclusion

RAG evaluation should answer more than "Was the final answer correct?"

It should prove:

- the authoritative source existed;
- the source was current and permitted;
- retrieval found it;
- ranking surfaced it;
- context preserved it;
- generation used it correctly;
- citations traced claims back to it;
- latency and cost remained acceptable;
- the evidence was complete enough for a release decision.

The engineering principle is:

> **Do not compress a multi-stage evidence system into one score when the real value of evaluation is knowing exactly where trust broke.**

---

## References

1. Ragas, **Faithfulness**, stable documentation: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
2. Ragas, **Context Recall**, stable documentation: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/
3. Ragas, **Available Metrics**, stable documentation: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
4. MLCommons, **Introducing the MLPerf End-to-End RAG Inference Benchmark**, 26 August 2026: https://mlcommons.org/2026/08/endtoend-inference/
5. NIST AI 600-1, **Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile**: https://doi.org/10.6028/NIST.AI.600-1
6. Ashok Kumar Manohar, **RAG Quality Engineering: A Practical Framework for Evaluating Retrieval-Augmented Generation Systems**: https://github.com/ashokmanohar-ai/rag-llm-evaluation-lab/blob/main/WHITEPAPER.md
7. Reference implementation: https://github.com/ashokmanohar-ai/rag-llm-evaluation-lab
