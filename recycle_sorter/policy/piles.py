from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class Piles:
    """Sorted piles as square zones, each holding a small grid of drop slots.

    A pile is a taught center (poses.yaml -> piles) plus the shared size and slot
    pitch from workspace.yaml. Successive items go to successive slots, so a pile
    spreads out instead of every item landing on the one before it.
    """

    def __init__(self, poses: dict[str, Any], workspace: dict[str, Any]):
        self.centers: dict[str, dict[str, float]] = poses.get("piles") or {}
        self.workspace = workspace
        self.size: float = workspace["piles"]["size"]
        self.pitch: float = workspace["piles"]["slot_pitch"]
        self.counts: dict[str, int] = {name: 0 for name in self.centers}

    @property
    def names(self) -> list[str]:
        return list(self.centers)

    def slots(self, name: str) -> list[tuple[float, float]]:
        c = self.centers[name]
        n = max(1, int(self.size // self.pitch))
        offsets = [(i - (n - 1) / 2) * self.pitch for i in range(n)]
        return [(c["x"] + dx, c["y"] + dy) for dx in offsets for dy in offsets]

    def next_slot(self, name: str) -> tuple[float, float]:
        if name not in self.centers:
            raise KeyError(f"pile {name!r} not taught yet - run scripts/01_teach_pose.py --pile {name}")
        slots = self.slots(name)
        i = self.counts[name]
        if i and i % len(slots) == 0:
            log.warning("pile %s is full (%d slots): stacking on earlier items", name, len(slots))
        self.counts[name] += 1
        return slots[i % len(slots)]

    def validate(self) -> None:
        """Fail before any motion if a pile could be re-picked or is out of reach of the bounds."""
        u, b = self.workspace["unsorted_zone"], self.workspace["bounds"]
        half = self.size / 2
        problems = []
        for name, c in self.centers.items():
            overlaps_x = c["x"] - half < u["x"][1] and c["x"] + half > u["x"][0]
            overlaps_y = c["y"] - half < u["y"][1] and c["y"] + half > u["y"][0]
            if overlaps_x and overlaps_y:
                problems.append(f"pile {name} overlaps the unsorted zone: sorted items would be picked again")
            for x, y in self.slots(name):
                if not (b["x"][0] <= x <= b["x"][1] and b["y"][0] <= y <= b["y"][1]):
                    problems.append(f"pile {name} slot ({x:.0f}, {y:.0f}) is outside the workspace bounds")
                    break
        if problems:
            raise ValueError("; ".join(problems))
