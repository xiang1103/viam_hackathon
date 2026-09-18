from __future__ import annotations

from typing import Any

from ..types import Classification
from .piles import REJECT


class SortPolicy:
    """Which pile does a classified object belong to?

    By default every class is its own pile. `groups` in sort.yaml merges classes
    ({warm: [red, orange, yellow]} -> one "warm" pile). Results below the mode's
    min_confidence go to reject rather than being guessed at.
    Where the piles physically go is PileLayout's job, not this class's.
    """

    def __init__(self, sort_cfg: dict[str, Any], mode: str):
        if mode not in sort_cfg["modes"]:
            raise ValueError(f"mode {mode!r} not in sort.yaml modes: {list(sort_cfg['modes'])}")
        m = sort_cfg["modes"][mode] or {}
        self.min_confidence: float = m.get("min_confidence", 0.0)
        self.group_of = {label: group for group, labels in (m.get("groups") or {}).items() for label in labels}

    def pile_key(self, c: Classification) -> str:
        if c.confidence < self.min_confidence:
            return REJECT
        return self.group_of.get(c.label, c.label)
