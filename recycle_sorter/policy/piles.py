from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

REJECT = "reject"


@dataclass
class PileZone:
    key: str  # the class (or group) this pile holds
    area: str
    rect: tuple[float, float, float, float]  # x0, x1, y0, y1
    slots: list[tuple[float, float]]
    pitch: float
    used: int = 0

    @property
    def full(self) -> bool:
        return self.used >= len(self.slots)


class PileLayout:
    """Creates sorted piles on demand inside the configured `sorted_areas`.

    Nothing is fixed in advance except where piles are ALLOWED to go. After the
    unsorted zone has been assessed, plan() carves one pile per class out of that
    space: slot count from how many items were seen, slot spacing from how big
    they are. Piles are packed along each area's long side.

    It stays dynamic afterwards, because a heap hides things: a class first seen
    mid-run gets a new pile from the remaining space, and a pile that fills up
    gets an extension.
    """

    def __init__(self, workspace: dict[str, Any], poses: dict[str, Any] | None = None):
        self.workspace = workspace
        self.cfg = workspace["piles"]
        # Areas taught at the table (poses.yaml) replace the guesses in workspace.yaml.
        self.areas: dict[str, dict[str, list[float]]] = (poses or {}).get("sorted_areas") or workspace["sorted_areas"]
        self.cursor = {name: 0.0 for name in self.areas}  # mm already used along each area's long side
        self.zones: dict[str, list[PileZone]] = {}

    # -- setup ---------------------------------------------------------------

    def validate(self) -> None:
        """Fail before any motion if an area could cause re-picking or leaves the safety bounds."""
        u, b = self.workspace["unsorted_zone"], self.workspace["bounds"]

        def overlap(p, q) -> bool:
            return all(p[k][0] < q[k][1] and p[k][1] > q[k][0] for k in ("x", "y"))

        problems = []
        names = list(self.areas)
        for i, name in enumerate(names):
            a = self.areas[name]
            if overlap(a, u):
                problems.append(f"sorted area {name} overlaps the unsorted zone: sorted items would be picked again")
            if not all(b[k][0] <= a[k][0] < a[k][1] <= b[k][1] for k in ("x", "y")):
                problems.append(f"sorted area {name} is outside the workspace bounds {b['x']} x {b['y']}")
            problems += [f"sorted areas {name} and {o} overlap" for o in names[i + 1 :] if overlap(a, self.areas[o])]
        if problems:
            raise ValueError("; ".join(problems))

    def plan(self, demand: dict[str, tuple[int, float]]) -> None:
        """demand: pile key -> (items seen, largest item length in mm).

        Reject is reserved first: whatever else runs out of room, there must be
        somewhere to put an item once it is in the gripper. Then the biggest classes,
        so they get the contiguous space. Classes that do not fit share reject.
        """
        # Reject can receive any class, so space it for the largest item seen.
        largest = max([size for _, size in demand.values()], default=0.0)
        demand = {REJECT: (0, largest), **demand}
        for key, (count, size) in sorted(demand.items(), key=lambda kv: (kv[0] != REJECT, -kv[1][0], kv[0])):
            if key not in self.zones:
                self._allocate(key, count + self.cfg["spare_slots"], size)
        if REJECT not in self.zones:
            raise ValueError("the sorted areas are too small for even a reject pile - enlarge sorted_areas")

    # -- allocation ----------------------------------------------------------

    def _allocate(self, key: str, capacity: int, item_size: float) -> PileZone | None:
        pitch = max(self.cfg["min_pitch"], item_size + self.cfg["slot_gap"])
        best: tuple[str, int, int] | None = None  # area, rows, cols
        for name, a in self.areas.items():
            long, short = ("x", "y") if a["x"][1] - a["x"][0] >= a["y"][1] - a["y"][0] else ("y", "x")
            rows = int((a[short][1] - a[short][0]) // pitch)
            if rows < 1:
                continue
            wanted = math.ceil(capacity / rows)
            cols = min(wanted, int((a[long][1] - a[long][0] - self.cursor[name]) // pitch))
            if cols < 1:
                continue
            if cols == wanted:  # first area that fits the whole pile wins
                best = (name, rows, cols)
                break
            if best is None or cols * rows > best[1] * best[2]:
                best = (name, rows, cols)
        if best is None:
            log.warning("no room left in the sorted areas for a %r pile", key)
            return None

        name, rows, cols = best
        a = self.areas[name]
        long, short = ("x", "y") if a["x"][1] - a["x"][0] >= a["y"][1] - a["y"][0] else ("y", "x")
        start = a[long][0] + self.cursor[name]
        mid = (a[short][0] + a[short][1]) / 2
        slots = []
        for i in range(cols):
            for j in range(rows):
                p = {long: start + (i + 0.5) * pitch, short: mid + (j - (rows - 1) / 2) * pitch}
                slots.append((p["x"], p["y"]))
        span = {long: (start, start + cols * pitch), short: (a[short][0], a[short][1])}
        self.cursor[name] += cols * pitch + self.cfg["pile_gap"]

        zone = PileZone(key, name, (*span["x"], *span["y"]), slots, pitch)
        self.zones.setdefault(key, []).append(zone)
        log.info("pile %r: %d slot(s) at %.0f mm pitch in area %s", key, len(slots), pitch, name)
        return zone

    def next_slot(self, key: str, item_size: float) -> tuple[str, tuple[float, float]]:
        """Returns (pile actually used, slot xy). Falls back to reject when no pile can be made."""
        for zone in self.zones.get(key, []):
            # An item larger than the pile was spaced for would crowd its neighbours.
            if not zone.full and item_size + self.cfg["slot_gap"] <= zone.pitch + 1e-6:
                return key, self._take(zone)
        # New class, full pile, or oversized item: try to make (more) room.
        zone = self._allocate(key, self.cfg["spare_slots"] + 1, item_size)
        if zone:
            return key, self._take(zone)
        if key != REJECT:
            if not self.zones.get(key):
                log.warning("sending %r item to reject", key)
                return self.next_slot(REJECT, item_size)
        zones = self.zones.get(key)
        if not zones:
            raise RuntimeError("the sorted areas are full and there is no reject pile - enlarge sorted_areas")
        zone = zones[-1]
        log.warning("pile %r is full and cannot grow: stacking on earlier items", key)
        return key, self._take(zone)

    @staticmethod
    def _take(zone: PileZone) -> tuple[float, float]:
        slot = zone.slots[zone.used % len(zone.slots)]
        zone.used += 1
        return slot

    def summary(self) -> dict[str, str]:
        return {k: f"{sum(z.used for z in zs)}/{sum(len(z.slots) for z in zs)} slots" for k, zs in self.zones.items()}
