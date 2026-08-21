"""Dependency-free JSON, HTML, and JUnit report writers."""

from __future__ import annotations

import html
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from rag_eval.models import EvaluationReport


def write_json(report: EvaluationReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return destination


def write_html(report: EvaluationReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(_format(value))}</td></tr>"
        for key, value in sorted(report.metrics.items())
    )
    case_rows = "".join(
        "<tr>"
        f"<td>{html.escape(case.case_id)}</td>"
        f"<td class='{str(case.passed).lower()}'>{'PASS' if case.passed else 'FAIL'}</td>"
        f"<td>{html.escape(', '.join(case.reason_codes) or '—')}</td>"
        "<td><details><summary>View</summary><pre>"
        f"{html.escape(json.dumps(case.diagnostics, indent=2))}"
        "</pre></details></td>"
        "</tr>"
        for case in report.cases
    )
    gate_items = (
        "".join(f"<li>{html.escape(item)}</li>" for item in report.gate_failures) or "<li>None</li>"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>RAG Evaluation {html.escape(report.evaluation_id)}</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:2rem;color:#172033;background:#f6f8fb}}
main{{max-width:1200px;margin:auto}}
table{{border-collapse:collapse;width:100%;background:white;margin:1rem 0}}
th,td{{padding:.65rem;border:1px solid #dfe4ec;text-align:left;vertical-align:top}}
th{{background:#14213d;color:white}}.true{{color:#087830;font-weight:700}}.false{{color:#b42318;font-weight:700}}
.gate{{display:inline-block;padding:.4rem .7rem;border-radius:.4rem}}
pre{{white-space:pre-wrap;max-width:70ch}}code{{background:#eef2f7;padding:.15rem .3rem}}
</style></head><body><main>
<h1>RAG &amp; LLM Evaluation Report</h1>
<p><span class="gate">QUALITY GATE: {report.quality_gate}</span></p>
<p>Dataset: <code>{html.escape(report.dataset)}</code></p>
<p>Evaluation: <code>{report.evaluation_id}</code></p>
<h2>Metrics</h2><table><tbody>{metric_rows}</tbody></table>
<h2>Gate findings</h2><ul>{gate_items}</ul>
<h2>Case diagnostics</h2>
<table><thead><tr><th>Case</th><th>Status</th><th>Reason</th><th>Evidence</th></tr></thead>
<tbody>{case_rows}</tbody></table>
</main></body></html>"""
    destination.write_text(document, encoding="utf-8")
    return destination


def write_junit(report: EvaluationReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        name="rag-evaluation",
        tests=str(len(report.cases)),
        failures=str(sum(not case.passed for case in report.cases)),
    )
    for case in report.cases:
        test_case = ET.SubElement(suite, "testcase", name=case.case_id, classname="rag.evaluation")
        if not case.passed:
            failure = ET.SubElement(test_case, "failure", message=",".join(case.reason_codes))
            failure.text = json.dumps(case.diagnostics, indent=2)
    ET.indent(suite)
    destination.write_text(ET.tostring(suite, encoding="unicode"), encoding="utf-8")
    return destination


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
