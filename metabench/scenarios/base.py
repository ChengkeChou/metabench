"""Scenario definitions: the ADEMP halves that are not the method.

A scenario fixes the data-generating mechanism and the estimand, and says
nothing about how the data will be analysed.  Keeping those apart is what lets
two adapters be compared on identical data, and what stops a scenario from
quietly encoding a method preference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def fingerprint(payload: dict[str, Any]) -> str:
    """Stable hash of a scenario or configuration, for the manifest."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Scenario:
    """One cell of the simulation grid.

    ``dgm`` holds the data-generating parameters, ``estimand`` says what the
    interval is supposed to cover -- ``mu`` for a confidence interval, the true
    effect of a *new* study for a prediction interval -- and ``truth`` is
    derived from ``dgm`` rather than stored, so a scenario cannot disagree with
    itself.
    """

    scenario_id: str
    family: str
    dgm: dict[str, Any]
    estimand: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return fingerprint(self.to_dict())

    def truth(self) -> float:
        """The value the interval under test is meant to cover."""
        return float(self.estimand.get("value", self.dgm.get("mu", 0.0)))
