from __future__ import annotations

from typing import Any

import numpy as np

from ..types import ObjectObservation


def choose_next(
    objects: list[ObjectObservation],
    workspace: dict[str, Any],
    blacklist: list[np.ndarray] | None = None,
    blacklist_radius: float = 40.0,
) -> ObjectObservation | None:
    """Pick the easiest object: graspable, topmost, and with the most room around it."""
    max_open = workspace["gripper"]["max_open"]
    candidates = [
        o
        for o in objects
        if (o.grasp_width or o.width) < max_open - 5
        and not any(np.linalg.norm(o.centroid - b) < blacklist_radius for b in (blacklist or []))
    ]
    if not candidates:
        return None
    # Height dominates so piles are taken apart from the top; isolation breaks ties.
    return max(candidates, key=lambda o: o.top_z + 0.2 * min(o.isolation, 150.0))
