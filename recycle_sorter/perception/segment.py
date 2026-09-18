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


def project_mask(points: np.ndarray, frame: Frame) -> np.ndarray:
    """World points -> HxW bool image mask, by projecting them back through the camera."""
    r, t = frame.cam_to_world[:3, :3], frame.cam_to_world[:3, 3]
    cam = (points - t) @ r  # inverse of a rigid transform, row-vector form
    k, (h, w) = frame.intrinsics, frame.color.shape[:2]
    front = cam[:, 2] > 1.0
    u = np.round(k.fx * cam[front, 0] / cam[front, 2] + k.cx).astype(int)
    v = np.round(k.fy * cam[front, 1] / cam[front, 2] + k.cy).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask = np.zeros((h, w), np.uint8)
    mask[v[inside], u[inside]] = 1
    # A point cloud is sparser than the image: close the gaps between projected points.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)


def observe(
    points: np.ndarray,
    frame: Frame,
    workspace: dict[str, Any],
    mask: np.ndarray | None = None,
    source_label: str | None = None,
) -> ObjectObservation | None:
    """Build one ObjectObservation from an object's WORLD-frame points (Nx3, mm).

    Shared by both perception paths, so size, top height and grasp angle mean the
    same thing whether the object came from Viam vision or from depth segmentation.
    """
    seg, table_top = workspace["segmentation"], workspace["table_top"]
    points = points[points[:, 2] > table_top + seg["min_height"]]  # drop any table points in the segment
    if len(points) < 10:
        return None
    z = points[:, 2]
    # The 95th percentile is the object's top even when the camera also sees one of its
    # sides, where a point-cloud CENTER would sit too low (the reason move_arm.py
    # hard-codes OBJECT_HEIGHT_MM).
    top_z = float(np.percentile(z, 95))

    # The top surface gives the footprint. Side faces seen in perspective would
    # otherwise drag the centroid away from the camera axis.
    top = points[z > top_z - seg["top_band"]]
    if len(top) < 10:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(top[:, :2].astype(np.float32))
    short_axis = angle if w <= h else angle + 90.0
    yaw = (short_axis + 90.0) % 180.0 - 90.0

    if mask is None:
        mask = project_mask(points, frame)
    ys, xs = np.nonzero(mask)
    if len(xs):
        bx, by, bw, bh = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    else:  # object is outside the color image: still sortable, just has no crop
        bx = by = bw = bh = 0
    bbox_depth = frame.depth[by : by + bh, bx : bx + bw]
    return ObjectObservation(
        mask=mask,
        bbox=(bx, by, bw, bh),
        crop=frame.color[by : by + bh, bx : bx + bw].copy(),
        centroid=np.array([cx, cy], dtype=float),
        top_z=top_z,
        height=top_z - table_top,
        yaw_deg=float(yaw),
        width=float(min(w, h)),
        length=float(max(w, h)),
        area_px=int(mask.sum()),
        hole_ratio=float(np.mean(bbox_depth == 0)) if bbox_depth.size else 0.0,
        points=points,
        source_label=source_label,
    )


def in_unsorted_zone(o: ObjectObservation, workspace: dict[str, Any]) -> bool:
    zone = workspace["unsorted_zone"]
    return zone["x"][0] < o.centroid[0] < zone["x"][1] and zone["y"][0] < o.centroid[1] < zone["y"][1]


def finish(objects: list[ObjectObservation], workspace: dict[str, Any]) -> list[ObjectObservation]:
    """Common last step: keep plausible objects inside the unsorted zone, then score isolation."""
    max_dim = workspace["segmentation"]["max_object_dim"]
    objects = [o for o in objects if in_unsorted_zone(o, workspace) and max(o.length, o.height) <= max_dim]
    for o in objects:
        others = [np.linalg.norm(o.centroid - p.centroid) for p in objects if p is not o]
        o.isolation = float(min(others)) if others else float("inf")
    return objects


def segment(frame: Frame, workspace: dict[str, Any]) -> list[ObjectObservation]:
    """OpenCV fallback: objects are blobs standing above the table plane inside the unsorted zone."""
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
    blobs = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)

    objects = []
    for i in range(1, n):
        if stats[i][4] < seg["min_area_px"]:
            continue
        m = labels == i
        o = observe(pts[m], frame, workspace, mask=m)
        if o is not None:
            objects.append(o)
    return finish(objects, workspace)


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
