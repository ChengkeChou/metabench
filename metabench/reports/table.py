"""Report tables generated from ``results/`` — never entered by hand."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Sequence


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def row_for(summary: dict[str, Any]) -> Sequence[str]:
    metrics = summary["metrics"]
    ci, pi, pi_t = metrics["ci"], metrics["pi"], metrics["pi_t"]
    rates = metrics["rates"]
    return [
        summary["scenario"]["scenario_id"],
        summary["adapter"],
        str(metrics["reps"]),
        f"{_pct(ci['estimate'])} ±{_num(1.96 * ci['mcse'], 3)}",
        f"{_pct(pi['estimate'])} ±{_num(1.96 * pi['mcse'], 3)}",
        f"{_pct(pi_t['estimate'])} ±{_num(1.96 * pi_t['mcse'], 3)}",
        _num(metrics["ci_width"], 3),
        _num(metrics["pi_width"], 3),
        _num(metrics["bias"]["estimate"], 4),
        _num(metrics["dispersion"]["se_ratio"], 3),
        _pct(rates.get("failure_rate")),
        _pct(rates.get("boundary_rate")),
    ]


HEADER = ("scenario", "adapter", "reps", "CI", "PI", "PI(t)", "CI width",
          "PI width", "bias", "SE ratio", "fail", "boundary")


def table(summaries: Sequence[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |",
             "|" + "|".join(["---"] * len(HEADER)) + "|"]
    for summary in summaries:
        lines.append("| " + " | ".join(row_for(summary)) + " |")
    return "\n".join(lines) + "\n"


def load(results: pathlib.Path, match: str | None = None) -> list[dict]:
    """The newest summary per (scenario, adapter) directory.

    Several runs of the same cell are expected -- a screening pass, then a
    formal one -- and the table should show the current answer, not every
    historical file.  The raw records stay on disk; nothing is deleted.
    """
    newest: dict[pathlib.Path, pathlib.Path] = {}
    for path in sorted(results.rglob("summary_*.json")):
        if match and match not in str(path):
            continue
        directory = path.parent
        if directory not in newest or path.name > newest[directory].name:
            newest[directory] = path
    return [json.loads(p.read_text(encoding="utf-8"))
            for _, p in sorted(newest.items())]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="render result tables")
    parser.add_argument("results", nargs="?", default="results")
    parser.add_argument("--match", default=None)
    args = parser.parse_args(argv)
    print(table(load(pathlib.Path(args.results), args.match)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
