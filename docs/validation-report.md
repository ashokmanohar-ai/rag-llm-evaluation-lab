# Validation report

Validated on 21 August 2026 with Python 3.12.13 in offline mode (`EMBEDDING_PROVIDER=hashing`, `LLM_PROVIDER=mock`). These results prove deterministic framework behaviour on the synthetic Acme corpus; they do not claim real-model or production-corpus quality.

## Result

| Check | Result | Evidence |
|---|---|---|
| Dependency resolution/install | PASS | Editable install resolved 39 packages in an isolated environment |
| Ruff lint | PASS | No findings |
| Ruff formatting | PASS | 123 files already formatted |
| Strict MyPy | PASS | 73 source/test files, no issues |
| Automated tests | PASS | 45 passed |
| Measured coverage | PASS | 85.15%, threshold 80% |
| Wheel packaging | PASS | `rag_llm_evaluation_lab-1.0.0-py3-none-any.whl` built |
| Configuration/workflow parsing | PASS | All YAML and JSONL files parsed |
| Index build | PASS | 25 traceable chunks built from 24 original documents plus one superseded version |
| CLI grounded query | PASS | Returned `25 days` with `HR-ANNUAL-LEAVE-C001` |
| Retrieval suite | PASS | 25/25; Recall@5 1.0; MRR 1.0 |
| QA suite, three runs/case | PASS | 25/25; completeness, groundedness and citation correctness 1.0 |
| Hallucination detector suite | PASS | 10/10 expected-positive/negative cases classified |
| Citation validator suite | PASS | 10/10 valid/invalid cases classified |
| Adversarial suite | PASS | 10/10 injection, version, region, distractor and missing-evidence cases |
| Baseline create/compare | PASS | Portable measured baseline created; quality deltas 0 |
| Benchmark | PASS | Dense, hybrid, reranked, top-k and chunk-size configurations executed |
| API smoke flow | PASS | Health, query, evaluation, status and HTML report endpoints |
| JSON/HTML/JUnit reports | PASS | Generated for every evaluation suite |
| Dockerfile/Compose static review | PASS | Non-root runtime, health check, read-only Compose service, offline defaults |
| Local Docker build | NOT RUN | Docker/Podman is not installed in the validation environment; GitHub Actions performs the container build |
| GitHub Actions CI | PASS | Validate and Docker jobs completed successfully on pull request commit `8e33d47` |
| GitHub Actions regression | PASS | Retrieval, QA, baseline comparison and artifact upload completed successfully |

## Negative validation

- Missed relevant evidence fails retrieval.
- Wrong facts and numeric contradictions fail correctness/groundedness.
- Unsupported claims trigger hallucination detection.
- Wrong, invented and missing citations are detected.
- A safe insufficient-evidence refusal passes.
- Prompt and citation injection inside a document do not change system behaviour.
- Superseded and out-of-scope region/access documents are excluded.
- A threshold breach produces a blocking `FAIL`; latency can be configured as `CONDITIONAL_PASS`.

## Measured offline baseline

The committed baseline was produced by the CLI, not hand-authored. On the controlled QA dataset it recorded Recall@5 1.0, MRR 1.0, groundedness 1.0, completeness 1.0, citation correctness 1.0, hallucination rate 0.0 and stability 1.0. Precision@5 is 0.2 because every case has one labelled relevant document and the evaluation deliberately returns five candidates; this illustrates why recall and precision must be interpreted together.

## Known limitations

- Deterministic hashing and extractive mock generation validate controls, not semantic model quality.
- `all-MiniLM-L6-v2`, FAISS, CrossEncoder and hosted provider paths require their optional extras and separate provider evaluations.
- Local Docker execution was unavailable; the equivalent container build completed successfully in GitHub Actions.
- The FastAPI evaluation registry is in memory; production requires durable jobs, authentication, rate limiting and tenant-aware storage.
- The corpus is intentionally small and synthetic. Results must be recalibrated on representative, reviewed enterprise datasets.

Overall validation status: **PASS — READY FOR REVIEW**.
