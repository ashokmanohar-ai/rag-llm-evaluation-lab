from rich.console import Console
from rich.table import Table

from rag_eval.models import EvaluationReport


def render(report: EvaluationReport) -> None:
    console = Console()
    table = Table(title="RAG EVALUATION")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in sorted(report.metrics.items()):
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)
    console.print(f"QUALITY GATE: [bold]{report.quality_gate}[/bold]")
