from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import Frame, ObjectObservation
from .segment import level, to_pixels

log = logging.getLogger(__name__)


@dataclass
class ScanView:
    """How one item found from above appears in the eye-level scan picture."""

    box: tuple[int, int, int, int] | None  # x, y, w, h in the scan image; None = not in view
    distance: float  # camera to item, mm
    hidden: float  # fraction of the box covered by items nearer the camera
    crop: np.ndarray | None = None

    @property
    def readable(self) -> bool:
        return self.crop is not None


def _box(o: ObjectObservation, scan: Frame, table_top: float, margin: float) -> tuple[int, int, int, int] | None:
    half = max(o.length, o.width) / 2
    cx, cy = o.centroid
    corners = [(cx + dx * half, cy + dy * half, z) for dx in (-1, 1) for dy in (-1, 1) for z in (table_top, o.top_z)]
    px = to_pixels(scan, corners)
    if px is None:
        return None  # behind the camera
    h, w = scan.color.shape[:2]
    x0, y0, x1, y1 = px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0, x1, y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None  # out of frame, or too small to read anything from
    return x0, y0, x1 - x0, y1 - y0


def views(
    objects: list[ObjectObservation],
    scan: Frame,
    survey: Frame,
    workspace: dict[str, Any],
) -> list[ScanView]:
    """Match items found in the top-down survey to crops of the eye-level scan picture.

    Positions come from the survey (a tall item stands far above the depth noise there);
    the scan picture is only used for what the item LOOKS like. Each item's known 3D box is
    projected into the scan picture and cut out. An item mostly covered by one nearer the
    camera is reported as hidden rather than cropped: a crop of it would show the wrong item.
    """
    cfg = workspace["scan"]
    level(scan, workspace)  # so projections land on the image despite the camera's z offset...
    if scan.table_plane is None:
        scan.table_plane = survey.table_plane  # ...a side view may see too little table to fit it

    eye = scan.cam_to_world[:3, 3]
    out = []
    for o in objects:
        mid = np.array([o.centroid[0], o.centroid[1], (workspace["table_top"] + o.top_z) / 2])
        out.append(ScanView(_box(o, scan, workspace["table_top"], cfg["crop_margin"]), float(np.linalg.norm(mid - eye)), 0.0))

    for i, v in enumerate(out):
        if v.box is None:
            continue
        x, y, w, h = v.box
        covered = np.zeros((h, w), bool)
        for j, n in enumerate(out):
            if j == i or n.box is None or n.distance >= v.distance:
                continue
            nx, ny, nw, nh = n.box
            covered[max(ny, y) - y : max(min(ny + nh, y + h) - y, 0), max(nx, x) - x : max(min(nx + nw, x + w) - x, 0)] = True
        v.hidden = float(covered.mean())
        if v.hidden <= cfg["max_hidden"]:
            v.crop = scan.color[y : y + h, x : x + w].copy()
    return out
