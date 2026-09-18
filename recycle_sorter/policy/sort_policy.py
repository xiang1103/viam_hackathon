from __future__ import annotations

import logging
from typing import Any

from ..types import Classification

log = logging.getLogger(__name__)


class SortPolicy:
    """Decides which pile a classified object goes to.

    Per mode (sort.yaml):
      map:   fixed label -> pile. Several labels may share a pile.
      auto:  any other label claims the next free pile the first time it is seen,
             so the piles form themselves around whatever classes turn up.
    Low-confidence results, and labels left over once the piles run out, go to
    the reject pile.
    """

    def __init__(self, sort_cfg: dict[str, Any], mode: str, pile_names: list[str]):
        if mode not in sort_cfg["modes"]:
            raise ValueError(f"mode {mode!r} not in sort.yaml modes: {list(sort_cfg['modes'])}")
        m = sort_cfg["modes"][mode] or {}
        self.reject: str = sort_cfg["reject_pile"]
        self.fixed: dict[str, str] = dict(m.get("map") or {})
        self.auto: bool = m.get("auto", True)
        self.min_confidence: float = m.get("min_confidence", 0.0)
        self.assigned: dict[str, str] = {}
        self.free = [p for p in pile_names if p != self.reject and p not in self.fixed.values()]

        unknown = set(self.fixed.values()) - set(pile_names)
        if unknown:
            raise ValueError(f"sort.yaml maps to piles that are not in poses.yaml: {sorted(unknown)}")

    def pile_for(self, c: Classification) -> str:
        if c.confidence < self.min_confidence:
            return self.reject
        if c.label in self.fixed:
            return self.fixed[c.label]
        if c.label not in self.assigned:
            if not (self.auto and self.free):
                return self.reject
            self.assigned[c.label] = self.free.pop(0)
            log.info("new class %r claims %s", c.label, self.assigned[c.label])
        return self.assigned[c.label]

    @property
    def legend(self) -> dict[str, str]:
        return {**self.fixed, **self.assigned}
