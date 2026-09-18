from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..types import Frame, ObjectObservation
from .frames import world_points


def table_height(frame: Frame) -> float:
    """Median world z of everything in view. Use on a BARE table to measure table_top."""
    pts, valid = world_points(frame)
    return float(np.median(pts[..., 2][valid]))


def segment(frame: Frame, workspace: dict[str, Any]) -> list[ObjectObservation]:
    """Find objects as blobs standing above the table plane inside the unsorted zone."""
    seg = workspace["segmentation"]
    roi = workspace["unsorted_zone"]
    table_top = workspace["table_top"]

    pts, valid = world_points(frame)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    above = (
        valid
        & (z > table_top + seg["min_height"])
        & (x > roi["x"][0]) & (x < roi["x"][1])
        & (y > roi["y"][0]) & (y < roi["y"][1])
    )

    mask = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    objects: list[ObjectObservation] = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < seg["min_area_px"]:
            continue
        m = labels == i
        top_z = float(np.percentile(z[m], 95))

        # The top surface gives the footprint. Side faces seen in perspective
        # would otherwise drag the centroid away from the camera axis.
        top = m & (z > top_z - seg["top_band"])
        xy = np.column_stack([x[top], y[top]]).astype(np.float32)
        if len(xy) < 10:
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(xy)
        short_axis = angle if w <= h else angle + 90.0
        yaw = (short_axis + 90.0) % 180.0 - 90.0

        bbox_depth = frame.depth[by : by + bh, bx : bx + bw]
        objects.append(
            ObjectObservation(
                mask=m,
                bbox=(int(bx), int(by), int(bw), int(bh)),
                crop=frame.color[by : by + bh, bx : bx + bw].copy(),
                centroid=np.array([cx, cy], dtype=float),
                top_z=top_z,
                height=top_z - table_top,
                yaw_deg=float(yaw),
                width=float(min(w, h)),
                length=float(max(w, h)),
                area_px=int(area),
                hole_ratio=float(np.mean(bbox_depth == 0)),
            )
        )

    for o in objects:
        others = [np.linalg.norm(o.centroid - p.centroid) for p in objects if p is not o]
        o.isolation = float(min(others)) if others else float("inf")
    return objects


def zone_outline_px(frame: Frame, workspace: dict[str, Any]) -> np.ndarray | None:
    """The unsorted zone's corners (on the table plane) projected into the image, or None if behind the camera."""
    zone, z = workspace["unsorted_zone"], workspace["table_top"]
    corners = [(zone["x"][i], zone["y"][j], z, 1.0) for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))]
    cam = (np.linalg.inv(frame.cam_to_world) @ np.array(corners).T).T
    if (cam[:, 2] <= 0).any():
        return None
    k = frame.intrinsics
    return np.column_stack([k.fx * cam[:, 0] / cam[:, 2] + k.cx, k.fy * cam[:, 1] / cam[:, 2] + k.cy]).round().astype(np.int32)


def draw(
    frame: Frame,
    objects: list[ObjectObservation],
    labels: list[str] | None = None,
    workspace: dict[str, Any] | None = None,
) -> np.ndarray:
    """Debug overlay: unsorted zone outline, contours, index, label, and metric size."""
    img = frame.color.copy()
    outline = zone_outline_px(frame, workspace) if workspace else None
    if outline is not None:
        cv2.polylines(img, [outline], isClosed=True, color=(255, 0, 255), thickness=2)
    for i, o in enumerate(objects):
        contours, _ = cv2.findContours(o.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        bx, by, _, _ = o.bbox
        text = f"{i}:{labels[i] if labels else ''} {o.width:.0f}x{o.length:.0f} h{o.height:.0f}"
        cv2.putText(img, text, (bx, max(by - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img
