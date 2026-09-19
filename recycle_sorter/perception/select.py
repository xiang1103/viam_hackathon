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
    """Pick the easiest object: graspable, topmost, and with the most room around it.

    Items the arm must not go to are left where they are, with a warning: further than max_reach
    from its base, or with a grasp target (the item's position plus the calibration correction
    gripper.xy_offset) outside the workspace bounds - e.g. a can at the table's edge.
    """
    from ..manipulation.pickplace import grasp_pose
    from ..manipulation.safety import UnsafeTarget, check_target

    max_open = workspace["gripper"]["max_open"]
    # Picks have their own limits (pick.max_reach / pick.y_min); piles keep the layout's. Reach is
    # measured to where the arm actually goes: the grasp point plus the calibration correction.
    reach = workspace["pick"].get("max_reach", workspace["sorted_layout"]["max_reach"])
    out_of_reach = [o for o in objects if np.hypot(*grasp_pose(o, workspace)[:2]) > reach]
    if out_of_reach:
        log.warning(
            "%d item(s) are further than %d mm from the arm base and will be left: %s",
            len(out_of_reach), reach, ", ".join(f"({o.centroid[0]:.0f}, {o.centroid[1]:.0f})" for o in out_of_reach),
        )
    out_of_bounds = []
    for o in objects:
        x, y, z, _ = grasp_pose(o, workspace)
        try:
            check_target(x, y, z + workspace["pick"]["approach"], workspace, workspace["pick"].get("y_min"))
        except UnsafeTarget as e:
            out_of_bounds.append(o)
            log.warning("item at (%.0f, %.0f) is left: its grasp target is outside the bounds (%s)",
                        o.centroid[0], o.centroid[1], e)
    candidates = [
        o
        for o in objects
        if (o.grasp_width or o.width) < max_open - 5
        and not any(o is far for far in out_of_reach)
        and not any(o is out for out in out_of_bounds)
        and not any(np.linalg.norm(o.centroid - b) < blacklist_radius for b in (blacklist or []))
    ]
    if not candidates:
        return None
    # Height dominates so piles are taken apart from the top; isolation breaks ties.
    return max(candidates, key=lambda o: o.top_z + 0.2 * min(o.isolation, 150.0))
