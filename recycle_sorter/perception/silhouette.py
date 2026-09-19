from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..types import Frame, ObjectObservation
from .segment import to_pixels

_ANGLES = np.linspace(0, 2 * np.pi, 24, endpoint=False)
_RING = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])


def is_upright(obs: ObjectObservation, workspace: dict[str, Any]) -> bool:
    """A standing round item (a can): tall, with a footprint the gripper can close around from any side."""
    g = workspace["gripper"]
    return obs.height >= g.get("upright_min_height", 80) and obs.length <= g["max_open"] + 10


def _outline(frame: Frame, x: float, y: float, r: float, top: float, table: float, cfg: dict[str, Any]) -> np.ndarray | None:
    """The picture outline of a can standing at (x, y): lid, shoulder and foot circles, hulled."""
    rings = [(r * cfg.get("lid_ratio", 0.8), top), (r, top - cfg.get("shoulder", 12.0)), (r, table)]
    pts = np.vstack([np.column_stack([x + _RING[:, 0] * rr, y + _RING[:, 1] * rr, np.full(len(_RING), z)]) for rr, z in rings])
    px = to_pixels(frame, pts)
    return None if px is None else cv2.convexHull(px)


def fit_upright(
    frame: Frame, mask: np.ndarray, start_xy: np.ndarray, top_z: float, workspace: dict[str, Any]
) -> tuple[np.ndarray, float, float] | None:
    """Where a standing can's axis is, from its SILHOUETTE in the colour picture: (xy, radius, IoU).

    Depth drops out on a shiny lid - half of it, a different half in every picture - and a centre
    taken from the depth points jumps with it (18 mm between two pictures of an untouched can,
    2026-09-19). The colour mask has no such holes. A can standing on the table is a solid of
    revolution about a vertical axis: that solid is projected into the picture and (x, y, radius)
    are moved until its outline covers the mask best. Depth only supplies the height. On the same
    frames the centre then moved 0.2-0.8 mm and sat 3 mm from a depth-free close-up of the lid.
    """
    cfg = workspace.get("silhouette") or {}
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    x0, y0 = max(int(xs.min()) - 30, 0), max(int(ys.min()) - 30, 0)
    m = mask[y0 : int(ys.max()) + 30, x0 : int(xs.max()) + 30]
    table = workspace["table_top"]
    r_min, r_max = cfg.get("radius", (20.0, 60.0))

    def score(p: np.ndarray) -> float:
        hull = _outline(frame, p[0], p[1], p[2], top_z, table, cfg)
        if hull is None:
            return 0.0
        canvas = np.zeros(m.shape, np.uint8)
        cv2.fillConvexPoly(canvas, hull - [x0, y0], 1)
        c = canvas.astype(bool)
        return float((c & m).sum() / max((c | m).sum(), 1))

    # Pattern search: try a step each way in x, y and radius; halve the steps when none helps.
    p = np.array([start_xy[0], start_xy[1], cfg.get("start_radius", 33.0)], dtype=float)
    best, step = score(p), np.array([8.0, 8.0, 3.0])
    while step[0] > 0.4:
        moved = False
        for i in range(3):
            for s in (1.0, -1.0):
                q = p.copy()
                q[i] += s * step[i]
                if r_min <= q[2] <= r_max and (v := score(q)) > best:
                    p, best, moved = q, v, True
        if not moved:
            step = step / 2
    return p[:2], float(p[2]), best
