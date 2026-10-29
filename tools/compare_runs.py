"""Two runs of the same grid, side by side.

A "before" and an "after" of the same cells are far more useful as one table
than as two, and the *differences* are what a change is judged on: a fix that
moves a coverage rate is doing something, a fix that moves the boundary rate of
a model whose likelihood puts it on the boundary is doing something else.

    python tools/compare_runs.py results results_v116
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

METRICS = (
    ("suboptimal", "prop",
     lambda m: m.get("proportions", {}).get("suboptimal", {}).get("estimate")),
    ("gap max", "num", lambda m: m.get("gap", {}).get("max")),
    ("non-PD", "prop", lambda m: m["rates"].get("non_pd_rate")),
    ("boundary", "prop", lambda m: m["rates"].get("boundary_rate")),
    ("multi-start", "prop", lambda m: m["rates"].get("multistart_conflict_rate")),
    ("Frob med", "num", lambda m: m.get("frobenius", {}).get("median")),
    ("min-eig", "num", lambda m: m.get("min_eig_ratio", {}).get("median")),
    ("runtime med", "sec", lambda m: m.get("runtime", {}).get("median")),
)


def _newest(root: pathlib.Path) -> dict[tuple[str, str], dict[str, Any]]:
    newest: dict[tuple[str, str], pathlib.Path] = {}
    for path in sorted(root.rglob("summary_*.json")):
        # <root>/<scenario>/<adapter family>/<method>/summary.json
        key = (path.parent.parent.parent.name, path.parent.name)
        if key not in newest or path.name > newest[key].name:
            newest[key] = path
    return {key: json.loads(p.read_text(encoding="utf-8"))
            for key, p in newest.items()}


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "prop":
        return f"{100 * value:.1f}%"
    if kind == "sec":
        return f"{value:.2f} s"
    return f"{value:.3f}"


def _versions(summaries: list[dict[str, Any]]) -> list[str]:
    seen = []
    for summary in summaries:
        version = summary.get("adapter_version")
        if version and version not in seen:
            seen.append(version)
    return seen


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__)
        return 2
    before = _newest(pathlib.Path(argv[0]))
    after = _newest(pathlib.Path(argv[1]))
    print(f"before: {argv[0]}   after: {argv[1]}")
    print("(" + ", ".join(sorted({k[1] for k in before | after})) + ")")
    header = ["scenario"] + [f"{name} (before → after)" for name, _, _ in METRICS]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for key in sorted(set(before) & set(after)):
        b, a = before[key]["metrics"], after[key]["metrics"]
        cells = [key[0].replace("covariance_", "")]
        for _, kind, get in METRICS:
            left, right = get(b), get(a)
            if left is None and right is None:
                cells.append("—")
            else:
                cells.append(f"{_fmt(left, kind)} → {_fmt(right, kind)}")
        print("| " + " | ".join(cells) + " |")
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))
    if only_before or only_after:
        print()
        print("not compared: before only", [k[0] for k in only_before],
              "after only", [k[0] for k in only_after])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
