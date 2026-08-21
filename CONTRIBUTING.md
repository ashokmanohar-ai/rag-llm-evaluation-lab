# Contributing

1. Create a feature branch from `main`.
2. Install with `python -m pip install -e '.[dev]'`.
3. Add or update controlled evaluation cases when behaviour changes.
4. Run `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest`.
5. Run the offline retrieval and QA evaluations and explain metric deltas in the pull request.

Do not update a baseline merely to make a regression pass. Baseline changes require a documented reason, the same dataset, measured output, and reviewer approval.

