from __future__ import annotations

import copy
import logging
import math
from typing import Any, Awaitable, Callable

import numpy as np

from ..types import ObjectObservation

log = logging.getLogger(__name__)

AUTO = "auto"
FindObjects = Callable[[dict[str, Any]], Awaitable[list[ObjectObservation]]]


def zone_around(objects: list[ObjectObservation], workspace: dict[str, Any]) -> dict[str, list[float]]:
    """The unsorted zone = the box around everything found, plus a margin, kept inside the search region.

    The margin leaves room for items that were hidden in the heap or get nudged while
    their neighbours are picked.
    """
    region, margin = workspace["search_region"], workspace["zone_margin"]
    xy = np.vstack([o.points[:, :2] if o.points is not None else o.centroid[None, :] for o in objects])
    return {
        "x": [max(region["x"][0], float(xy[:, 0].min()) - margin), min(region["x"][1], float(xy[:, 0].max()) + margin)],
        "y": [max(region["y"][0], float(xy[:, 1].min()) - margin), min(region["y"][1], float(xy[:, 1].max()) + margin)],
    }


def _strips(zone, workspace, toward_minus_y: bool, prefix: str) -> dict[str, dict[str, list[float]]]:
    """Strips filling the table between one y-side of the zone and the bounds on that side."""
    cfg, bounds = workspace["sorted_layout"], workspace["bounds"]
    start = zone["y"][0] if toward_minus_y else zone["y"][1]
    limit = bounds["y"][0] if toward_minus_y else bounds["y"][1]
    span = (start - limit if toward_minus_y else limit - start) - cfg["gap"]
    n = int(span // cfg["strip_depth"])
    if n < 1:
        return {}
    depth, sign, areas = span / n, (-1.0 if toward_minus_y else 1.0), {}
    for i in range(n):
        near = start + sign * (cfg["gap"] + i * depth)
        far = near + sign * (depth - cfg["strip_gap"])
        x_far = min(cfg["x"][1], math.sqrt(max(cfg["max_reach"] ** 2 - max(abs(near), abs(far)) ** 2, 0.0)))
        if x_far - cfg["x"][0] >= 100:  # else: out of reach this far out
            areas[f"{prefix}_{i + 1}"] = {"x": [cfg["x"][0], round(x_far, 1)], "y": sorted([round(near, 1), round(far, 1)])}
    return areas


def areas_beyond(zone: dict[str, list[float]], workspace: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Sorted areas = the free table around the unsorted zone.

    - `strip_N`: between the zone and the bounds on the open side (sorted_layout.open_side),
    - `edge_N`:  the same on the other side, where the table is narrower,
    - `front`:   beyond the zone, further from the arm, across the zone's own width.
    Every area is cut short so that its outermost corner stays within the arm's reach.
    """
    cfg, bounds = workspace["sorted_layout"], workspace["bounds"]
    open_minus_y = cfg["open_side"] == "-y"
    areas = {**_strips(zone, workspace, open_minus_y, "strip"), **_strips(zone, workspace, not open_minus_y, "edge")}

    # The table beyond the zone. It cannot collide with the strips: those start past the
    # zone's y edges, this stays within the zone's own y range.
    y0, y1 = max(zone["y"][0], bounds["y"][0]), min(zone["y"][1], bounds["y"][1])
    x0 = zone["x"][1] + cfg["gap"]
    x1 = min(cfg["front_x_max"], bounds["x"][1], math.sqrt(max(cfg["max_reach"] ** 2 - max(abs(y0), abs(y1)) ** 2, 0.0)))
    if x1 - x0 >= cfg["strip_depth"] * 0.75 and y1 - y0 >= 100:
        areas["front"] = {"x": [round(x0, 1), round(x1, 1)], "y": [round(y0, 1), round(y1, 1)]}

    if not areas:
        z = zone
        raise ValueError(
            f"no room for sorted piles: the unsorted items span x {z['x'][0]:.0f}..{z['x'][1]:.0f}, "
            f"y {z['y'][0]:.0f}..{z['y'][1]:.0f} and leave no free strip of {cfg['strip_depth']} mm inside the "
            f"reachable table (y {bounds['y'][0]}..{bounds['y'][1]}, within {cfg['max_reach']} mm of the arm). "
            "Group the starting items closer together."
        )
    return areas


async def resolve(workspace: dict[str, Any], find_objects: FindObjects) -> tuple[dict[str, Any], list[ObjectObservation] | None]:
    """Turn `unsorted_zone: auto` / `sorted_areas: auto` into concrete rectangles for this run.

    Returns (resolved workspace, objects found during the first look or None if no look was needed).
    With a fixed zone in the config nothing is detected here and the workspace is only copied.
    """
    ws = copy.deepcopy(workspace)
    objects = None
    if ws["unsorted_zone"] == AUTO:
        ws["unsorted_zone"] = copy.deepcopy(ws["search_region"])  # first look: anywhere it is allowed to be
        objects = await find_objects(ws)
        if objects:
            ws["unsorted_zone"] = zone_around(objects, ws)
            z = ws["unsorted_zone"]
            log.info("unsorted zone set around %d item(s): x %.0f..%.0f, y %.0f..%.0f", len(objects), *z["x"], *z["y"])
    if ws["sorted_areas"] == AUTO:
        if objects is not None and not objects:
            ws["sorted_areas"] = {}  # an empty table needs no piles (and its "zone" is the whole search region)
        else:
            ws["sorted_areas"] = areas_beyond(ws["unsorted_zone"], ws)
        for name, a in ws["sorted_areas"].items():
            log.info("sorted area %s: x %.0f..%.0f, y %.0f..%.0f", name, *a["x"], *a["y"])
    return ws, objects
