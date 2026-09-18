from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..types import Frame, ObjectObservation
from .frames import world_points


log = logging.getLogger(__name__)

MISALIGNED_TILT_DEG = 3.0


def fit_table(pts: np.ndarray, valid: np.ndarray, workspace: dict[str, Any]) -> tuple[np.ndarray, float] | None:
    """Fit the table plane z = a*x + b*y + d to what the camera actually sees.

    Returns ((a, b, d), tilt in degrees), or None when too little table is in view.
    Trimmed least squares: objects and noise are rejected as outliers over a few rounds.
    """
    b = workspace["bounds"]
    q = pts[::4, ::4][valid[::4, ::4]]
    # Loose in z on purpose: the camera's calibration can be tens of mm off.
    q = q[(q[:, 0] > b["x"][0]) & (q[:, 0] < b["x"][1]) & (q[:, 1] > b["y"][0]) & (q[:, 1] < b["y"][1])
          & (np.abs(q[:, 2] - workspace["table_top"]) < 150)]
    if len(q) < 500:
        return None
    for _ in range(6):
        a = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
        coef, *_ = np.linalg.lstsq(a, q[:, 2], rcond=None)
        r = q[:, 2] - a @ coef
        q = q[np.abs(r) < max(3.0, 1.5 * np.std(r))]
        if len(q) < 500:
            return None
    return coef, float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))


def level(frame: Frame, workspace: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float | None]:
    """World points with z re-expressed as table_top + HEIGHT ABOVE THE TABLE SEEN IN THIS FRAME.

    Heights relative to the fitted table survive a camera calibration that is a few cm
    or degrees off, and keep grasp heights tied to the arm's own table_top rather than
    to where the camera believes the table is. Returns (points, valid, tilt or None).
    """
    pts, valid = world_points(frame)
    fit = fit_table(pts, valid, workspace)
    if fit is None:
        return pts, valid, None
    (a, b, d), tilt = fit
    pts = pts.copy()
    pts[..., 2] = workspace["table_top"] + (pts[..., 2] - (a * pts[..., 0] + b * pts[..., 1] + d)) * np.cos(np.radians(tilt))
    if tilt > MISALIGNED_TILT_DEG:
        log.warning(
            "the table appears tilted %.1f deg. A RealSense whose depth is not aligned to color does this "
            "(depth has a wider lens than the color intrinsics used here): set align_color_depth: true on "
            "the camera. Positions and colors are unreliable until then.", tilt)
    return pts, valid, tilt


def table_report(frame: Frame, workspace: dict[str, Any]) -> str:
    """For a BARE table: where the camera sees the table, and whether depth looks aligned."""
    pts, valid = world_points(frame)
    fit = fit_table(pts, valid, workspace)
    if fit is None:
        return "could not find the table in view"
    (a, b, d), tilt = fit
    u = workspace["unsorted_zone"]
    cx, cy = sum(u["x"]) / 2, sum(u["y"]) / 2
    verdict = "OK" if tilt <= MISALIGNED_TILT_DEG else "NOT OK - set align_color_depth: true on the camera"
    return f"camera sees the table at z = {a * cx + b * cy + d:.1f} mm (zone center), tilt {tilt:.1f} deg: {verdict}"


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
    """OpenCV path: objects stand above the table inside the unsorted zone AND do not look like it.

    Depth alone is not enough: on a plain white table the RealSense produces noise bumps
    10-30 mm tall, the same height as a block. So a blob must also differ from the table's
    own color, unless it is clearly taller than any noise (a white item on a white table).
    """
    seg = workspace["segmentation"]
    roi = workspace["unsorted_zone"]
    table_top = workspace["table_top"]
    pad = seg["max_object_dim"]

    pts, valid, _ = level(frame, workspace)
    x, y, height = pts[..., 0], pts[..., 1], pts[..., 2] - table_top

    lab = cv2.cvtColor(frame.color, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = valid & (np.abs(height) < 4.0)
    table_color = np.median(lab[flat], axis=0) if flat.sum() > 100 else np.median(lab.reshape(-1, 3), axis=0)
    unlike_table = np.linalg.norm(lab - table_color, axis=2) > seg["color_delta"]

    above = (
        valid
        & (height > seg["min_height"])
        & (unlike_table | (height > seg["tall_height"]))
        # Padded, so an item straddling the zone edge is measured whole, not as a sliver.
        # finish() then keeps the items whose CENTER is inside the zone.
        & (x > roi["x"][0] - pad) & (x < roi["x"][1] + pad)
        & (y > roi["y"][0] - pad) & (y < roi["y"][1] + pad)
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
