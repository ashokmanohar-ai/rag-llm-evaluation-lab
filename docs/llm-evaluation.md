# LLM evaluation

Correctness asks whether claims are true. Relevance asks whether the response addresses the question. Completeness measures required fact coverage. Groundedness asks whether claims follow from supplied evidence. Faithfulness is used here as the same evidence-preservation family, while hallucination identifies unsupported, invented or contradictory claims.

Controlled facts, forbidden claims, refusal behaviour and citations are deterministic. A model judge is appropriate for semantic relevance or explanation quality only after its prompt, output schema, calibration set and version are controlled. Judge scores cannot override a wrong fact, invented citation or unsupported claim.

The mock provider extracts evidence sentences so CI can validate orchestration and gates. Its scores are not a claim about a real model. Provider evaluations must record model/deployment, prompt version, sampling settings, tokens, cost and repeated-run stability.

