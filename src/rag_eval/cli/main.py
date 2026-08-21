"""`rag-eval` command-line interface."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console

from rag_eval.config import load_rag_config, load_yaml, project_path
from rag_eval.evaluation import EvaluationRunner
from rag_eval.models import EvaluationReport
from rag_eval.pipeline import RAGPipeline
from rag_eval.regression import compare_metrics, load_baseline, save_baseline
from rag_eval.reporting import write_html, write_json, write_junit
from rag_eval.reporting.console import render

app = typer.Typer(help="Build and evaluate an offline-first RAG system.", no_args_is_help=True)
index_app = typer.Typer(help="Index lifecycle commands.")
baseline_app = typer.Typer(help="Regression baseline commands.")
app.add_typer(index_app, name="index")
app.add_typer(baseline_app, name="baseline")
console = Console()


def _pipeline(config_path: str = "config/rag.yaml") -> RAGPipeline:
    return RAGPipeline(project_path("data/knowledge"), load_rag_config(config_path))


def _run(dataset: str, runs_per_case: int = 1) -> EvaluationReport:
    pipeline = _pipeline()
    runner = EvaluationRunner(pipeline, load_yaml("config/quality-gates.yaml"))
    return runner.run(project_path(dataset), runs_per_case)


@index_app.command("build")
def index_build(
    output: Annotated[str, typer.Option(help="Index manifest path.")] = ".rag_eval/index.json",
) -> None:
    pipeline = _pipeline()
    destination = project_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": [chunk.model_dump(mode="json") for chunk in pipeline.chunks],
        "chunk_count": len(pipeline.chunks),
        "configuration": pipeline.config,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"Built index manifest with {len(pipeline.chunks)} chunks: {destination}")


@app.command("query")
def query(
    question: Annotated[str, typer.Argument(help="Question to answer from Acme knowledge.")],
    top_k: Annotated[int, typer.Option(min=1, max=50)] = 5,
) -> None:
    response = _pipeline().query(question, top_k=top_k)
    typer.echo(response.model_dump_json(indent=2))


@app.command("run")
def run(
    dataset: Annotated[
        str, typer.Option(help="JSONL evaluation dataset.")
    ] = "data/evaluation/qa.jsonl",
    runs_per_case: Annotated[int, typer.Option(min=1, max=10)] = 1,
    report_dir: Annotated[str, typer.Option(help="Report output directory.")] = "reports",
) -> None:
    report = _run(dataset, runs_per_case)
    output = project_path(report_dir)
    write_json(report, output / "evaluation.json")
    write_html(report, output / "evaluation.html")
    write_junit(report, output / "junit.xml")
    failures = [case.model_dump(mode="json") for case in report.cases if not case.passed]
    (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    render(report)
    if report.quality_gate == "FAIL":
        raise typer.Exit(1)


@baseline_app.command("create")
def baseline_create(
    dataset: Annotated[str, typer.Option()] = "data/evaluation/qa.jsonl",
    output: Annotated[str, typer.Option()] = "data/baselines/qa-v1.json",
) -> None:
    report = _run(dataset)
    save_baseline(
        project_path(output),
        report.metrics,
        dataset=dataset,
        configuration=report.configuration,
    )
    console.print(f"Baseline created from measured results: {project_path(output)}")


@baseline_app.command("compare")
def baseline_compare(
    dataset: Annotated[str, typer.Option()] = "data/evaluation/qa.jsonl",
    baseline: Annotated[str, typer.Option()] = "data/baselines/qa-v1.json",
) -> None:
    report = _run(dataset)
    comparison = compare_metrics(
        load_baseline(project_path(baseline)),
        report.metrics,
        expected_dataset=dataset,
    )
    console.print_json(data=comparison)


@app.command("benchmark")
def benchmark() -> None:
    """Compare dense, hybrid, top-k, and reranking configurations on one dataset."""
    base = load_rag_config()
    rows = []
    for name, mode, reranking, top_k, chunk_size, overlap in [
        ("dense-k5", "dense", False, 5, 600, 100),
        ("hybrid-k5", "hybrid", False, 5, 600, 100),
        ("hybrid-rerank-k3", "hybrid", True, 3, 600, 100),
        ("hybrid-rerank-k5", "hybrid", True, 5, 600, 100),
        ("hybrid-rerank-k10", "hybrid", True, 10, 600, 100),
        ("chunk-400-50", "hybrid", True, 5, 400, 50),
        ("chunk-800-150", "hybrid", True, 5, 800, 150),
    ]:
        config = json.loads(json.dumps(base))
        config["retrieval"]["mode"] = mode
        config["retrieval"]["final_top_k"] = top_k
        config["reranking"]["enabled"] = reranking
        config["chunking"]["chunk_size"] = chunk_size
        config["chunking"]["chunk_overlap"] = overlap
        pipeline = RAGPipeline(project_path("data/knowledge"), config)
        report = EvaluationRunner(pipeline, load_yaml("config/quality-gates.yaml")).run(
            project_path("data/evaluation/retrieval.jsonl")
        )
        rows.append(
            {
                "configuration": name,
                "mode": mode,
                "reranking": reranking,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "recall_at_5": report.metrics.get("recall_at_5"),
                "precision_at_5": report.metrics.get("precision_at_5"),
                "mrr": report.metrics.get("mrr"),
            }
        )
    console.print_json(data=rows)


if __name__ == "__main__":
    app()
