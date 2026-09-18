from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(eq=False)  # holds arrays: compare by identity, never element-wise
class Frame:
    """One RGB-D snapshot plus everything needed to interpret it offline."""

    color: np.ndarray  # HxWx3 BGR uint8
    depth: np.ndarray  # HxW uint16, millimeters, 0 = no reading
    intrinsics: Intrinsics
    cam_to_world: np.ndarray  # 4x4, camera frame -> world frame, mm
    joints: list[float] | None = None
    timestamp: str = ""
    # Set by perception: (a, b, d, table_top) of the table plane z = a*x + b*y + d as the CAMERA
    # sees it. Lets drawings convert the arm-referenced heights back to what the image shows.
    table_plane: tuple[float, float, float, float] | None = None


@dataclass(eq=False)  # holds arrays: compare by identity, never element-wise
class ObjectObservation:
    """One segmented object. All metric quantities are world frame, mm / degrees."""

    mask: np.ndarray  # HxW bool
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels
    crop: np.ndarray  # BGR crop of bbox
    centroid: np.ndarray  # (x, y) of the top surface
    top_z: float
    height: float  # above table
    yaw_deg: float  # direction of the SHORT axis (gripper closing direction), in [-90, 90)
    width: float  # short side
    length: float  # long side
    area_px: int
    hole_ratio: float  # fraction of bbox pixels with no depth reading
    isolation: float = float("inf")  # distance to nearest other object centroid
    # Where the fingers should close: a spot on the footprint that is solid all the way across.
    # For a plain block this is the centroid; for an arch or an L it is not.
    grasp_xy: np.ndarray | None = None
    grasp_width: float = 0.0  # how wide the object is along the closing line at grasp_xy
    points: np.ndarray | None = None  # Nx3 world-frame points, kept so a frame can be replayed offline
    source_label: str | None = None  # class given by the detector / vision service that found it, if any
    detection_confidence: float = 1.0  # the detector's confidence, when one found it


@dataclass
class Classification:
    label: str
    confidence: float
    meta: dict[str, Any] = field(default_factory=dict)
