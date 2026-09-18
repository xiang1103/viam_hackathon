from __future__ import annotations

from typing import Awaitable, Callable

import numpy as np

from ..types import Frame, Intrinsics


def deproject(depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Depth image (mm) -> HxWx3 points in the camera frame (x right, y down, z forward)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.dstack([x, y, z])


def to_world(points: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 3) points."""
    return points @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def world_points(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (HxWx3 world points, HxW valid mask)."""
    return to_world(deproject(frame.depth, frame.intrinsics), frame.cam_to_world), frame.depth > 0


async def cam_to_world_matrix(
    transform_point: Callable[[float, float, float], Awaitable[tuple[float, float, float]]],
) -> np.ndarray:
    """Build the camera->world matrix numerically from four point transforms.

    Asking the frame system where the camera origin and three axis points land
    sidesteps any orientation-vector convention mistakes.
    """
    d = 100.0
    origin = np.array(await transform_point(0, 0, 0))
    axes = [np.array(await transform_point(*p)) for p in ((d, 0, 0), (0, d, 0), (0, 0, d))]
    t = np.eye(4)
    for i, a in enumerate(axes):
        t[:3, i] = (a - origin) / d
    t[:3, 3] = origin
    return t
