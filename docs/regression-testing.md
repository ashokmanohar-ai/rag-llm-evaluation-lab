# RAG regression testing

Chunk size, overlap, embedding model, top-k, fusion, reranking, prompt or LLM changes can alter quality. Each candidate is evaluated on the same versioned dataset. Baselines store measured metrics, dataset path, configuration and creation time.

The comparator rejects a mismatched dataset. Quality gates combine absolute minima with allowable regression. Retrieval recall, groundedness, hallucination and citation accuracy normally block. Latency may be a conditional pass where the service owner accepts the trade-off.

Do not bless a regression by overwriting the baseline. Diagnose failed cases, explain the trade-off, review data-label changes, and update the baseline only after an intentional release decision.

