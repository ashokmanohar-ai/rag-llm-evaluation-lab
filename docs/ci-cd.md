# CI/CD

Normal pull-request CI requires no model credentials. It installs the core and development extras, runs Ruff, formatting, strict MyPy, all test categories, an offline index/query smoke flow and a container build.

The regression workflow runs retrieval and QA datasets, compares the measured baseline and uploads reports even when a gate fails. The benchmark workflow is manual because configuration sweeps are slower and should not add noise to every pull request.

Real-model evaluation belongs in a protected environment with scoped secrets, cost limits and explicit approval. It should complement, not replace, deterministic PR gates.

