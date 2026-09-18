from __future__ import annotations

from typing import Any

from ..types import Classification


class SortPolicy:
    """label -> bin name. Low-confidence or unmapped labels go to the unknown bin."""

    def __init__(self, sort_cfg: dict[str, Any], mode: str, min_confidence: float = 0.0):
        if mode not in sort_cfg["modes"]:
            raise ValueError(f"mode {mode!r} not in sort.yaml modes: {list(sort_cfg['modes'])}")
        self.table: dict[str, str] = sort_cfg["modes"][mode]
        self.unknown: str = sort_cfg["unknown_bin"]
        self.min_confidence = min_confidence

    def bin_for(self, c: Classification) -> str:
        if c.confidence < self.min_confidence:
            return self.unknown
        return self.table.get(c.label, self.unknown)
