"""Regenerate all standard reports from the QA dataset."""

from rag_eval.cli.main import run

if __name__ == "__main__":
    run(dataset="data/evaluation/qa.jsonl", report_dir="reports")
