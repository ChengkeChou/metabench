"""Report tables generated from ``results/`` — never entered by hand.

Each family has its own columns, because each family's estimand is a different
kind of object: a scalar interval, an ellipsoid, or a matrix.  Forcing them into
one header would mean reporting the covariance family's "coverage" as an empty
cell, and an empty cell reads as "fine".
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Sequence

FAMILY_ORDER = ("univariate", "multivariate", "covariance")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _gap(value: float | None) -> str:
    """A log-likelihood difference, with anything below noise shown as zero."""
    if value is None:
        return "—"
    return "0.00" if abs(value) < 5e-3 else f"{value:.2f}"


def _family(summary: dict[str, Any]) -> str:
    return summary.get("scenario", {}).get("family", "univariate")


def row_for(summary: dict[str, Any]) -> list[str]:
    metrics = summary["metrics"]
    rates = metrics["rates"]
    family = _family(summary)
    if family == "covariance":
        gap = metrics.get("gap", {})
        frob = metrics.get("frobenius", {})
        eig = metrics.get("min_eig_ratio", {})
        margin = metrics.get("proportions", {}).get("suboptimal", {})
        mcse = margin.get("mcse")
        return [
            summary["scenario"]["scenario_id"],
            summary["adapter"],
            str(metrics["reps"]),
            str(metrics.get("usable", metrics["reps"])),
            f"{_pct(margin.get('estimate'))} "
            f"±{_num(1.96 * mcse, 3) if mcse is not None else '—'}",
            _gap(gap.get("median")),
            _gap(gap.get("max")),
            _pct(rates.get("non_pd_rate")),
            _pct(rates.get("boundary_rate")),
            _pct(rates.get("multistart_conflict_rate")),
            _num(frob.get("median"), 4),
            _num(eig.get("median"), 3),
            _num(metrics.get("tau2_relative_bias"), 3),
            _num(metrics.get("rho_bias"), 3),
        ]
    if "joint_new" in metrics:                       # multivariate region
        joint, mean = metrics["joint_new"], metrics["joint_mean"]
        marginal = metrics["marginal_new"]
        per_outcome = " / ".join(_pct(m["estimate"]) for m in marginal)
        return [
            summary["scenario"]["scenario_id"],
            summary["adapter"],
            str(metrics["reps"]),
            f"{_pct(joint['estimate'])} ±{_num(1.96 * joint['mcse'], 3)}",
            f"{_pct(mean['estimate'])} ±{_num(1.96 * mean['mcse'], 3)}",
            per_outcome,
            _num(metrics["volume"], 4),
            "—",
            "—",
            "—",
            _pct(rates.get("failure_rate")),
            _pct(rates.get("boundary_rate")),
        ]
    ci, pi, pi_t = metrics["ci"], metrics["pi"], metrics["pi_t"]
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


HEADERS: dict[str, tuple[str, ...]] = {
    "univariate": ("scenario", "adapter", "reps", "CI", "PI", "PI(t)",
                   "CI width", "PI width", "bias", "SE ratio", "fail",
                   "boundary"),
    "multivariate": ("scenario", "adapter", "reps", "joint (new)",
                     "joint (mean)", "marginal (CI / PI)",
                     "CI width / volume", "PI width / —", "bias / —",
                     "SE ratio / —", "fail", "boundary"),
    "covariance": ("scenario", "adapter", "reps", "usable", "suboptimal",
                   "gap med", "gap max", "non-PD", "boundary", "multi-start",
                   "Frob med", "min-eig", "tau2 bias", "rho bias"),
}


def table(summaries: Sequence[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for family in FAMILY_ORDER:
        rows = [s for s in summaries if _family(s) == family]
        if not rows:
            continue
        header = HEADERS[family]
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join(["---"] * len(header)) + "|"]
        for summary in rows:
            lines.append("| " + " | ".join(row_for(summary)) + " |")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


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
