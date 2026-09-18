from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..types import ObjectObservation

log = logging.getLogger(__name__)


def choose_next(
    objects: list[ObjectObservation],
    workspace: dict[str, Any],
    blacklist: list[np.ndarray] | None = None,
    blacklist_radius: float = 40.0,
) -> ObjectObservation | None:
    """Pick the easiest object: graspable, topmost, and with the most room around it."""
    max_open = workspace["gripper"]["max_open"]
    reach = workspace["sorted_layout"]["max_reach"]
    out_of_reach = [o for o in objects if np.hypot(*(o.grasp_xy if o.grasp_xy is not None else o.centroid)) > reach]
    if out_of_reach:
        log.warning(
            "%d item(s) are further than %d mm from the arm base and will be left: %s",
            len(out_of_reach), reach, ", ".join(f"({o.centroid[0]:.0f}, {o.centroid[1]:.0f})" for o in out_of_reach),
        )
    candidates = [
        o
        for o in objects
        if (o.grasp_width or o.width) < max_open - 5
        and not any(o is far for far in out_of_reach)
        and not any(np.linalg.norm(o.centroid - b) < blacklist_radius for b in (blacklist or []))
    ]
    if not candidates:
        return None
    # Height dominates so piles are taken apart from the top; isolation breaks ties.
    return max(candidates, key=lambda o: o.top_z + 0.2 * min(o.isolation, 150.0))
