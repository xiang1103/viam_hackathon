"""Synthetic top-down RGB-D scene: a camera 400 mm above the table looking straight down."""
from __future__ import annotations

import cv2
import numpy as np

from recycle_sorter.types import Frame, Intrinsics

TABLE_TOP = 0.0
CAM_POS = np.array([400.0, 0.0, TABLE_TOP + 400.0])
INTR = Intrinsics(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)

# camera x -> world +x, camera y -> world -y, camera z -> world -z (looking down)
CAM_TO_WORLD = np.eye(4)
CAM_TO_WORLD[:3, :3] = np.diag([1.0, -1.0, -1.0])
CAM_TO_WORLD[:3, 3] = CAM_POS

BGR = {
    "red": (30, 30, 220),
    "green": (40, 180, 40),
    "blue": (220, 60, 30),
    "yellow": (30, 220, 230),
    "grey": (128, 128, 128),
}


def make_frame(blocks: list[dict]) -> Frame:
    """blocks: dicts with x, y (world mm), l, w, h (mm), yaw (deg, direction of the long side), color."""
    color = np.full((INTR.height, INTR.width, 3), (200, 205, 210), np.uint8)
    depth = np.full((INTR.height, INTR.width), 400, np.uint16)
    world_to_cam = np.linalg.inv(CAM_TO_WORLD)

    for b in sorted(blocks, key=lambda b: b["h"]):
        c, s = np.cos(np.deg2rad(b["yaw"])), np.sin(np.deg2rad(b["yaw"]))
        corners = []
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            lx, ly = dx * b["l"] / 2, dy * b["w"] / 2
            world = np.array([b["x"] + lx * c - ly * s, b["y"] + lx * s + ly * c, TABLE_TOP + b["h"], 1.0])
            xc, yc, zc = (world_to_cam @ world)[:3]
            corners.append([INTR.fx * xc / zc + INTR.cx, INTR.fy * yc / zc + INTR.cy])
        poly = np.round(np.array(corners)).astype(np.int32)
        cv2.fillPoly(color, [poly], BGR[b["color"]])
        cv2.fillPoly(depth, [poly], int(400 - b["h"]))

    return Frame(color=color, depth=depth, intrinsics=INTR, cam_to_world=CAM_TO_WORLD.copy())


SCENE = [
    {"x": 400, "y": 0, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "red"},
    {"x": 330, "y": -60, "l": 60, "w": 25, "h": 25, "yaw": 30, "color": "blue"},
    {"x": 470, "y": 70, "l": 30, "w": 30, "h": 50, "yaw": 0, "color": "green"},
    {"x": 460, "y": -80, "l": 40, "w": 40, "h": 20, "yaw": 0, "color": "grey"},
    # Outside the pile ROI (x > 550): an already-sorted item that must be ignored.
    {"x": 580, "y": 0, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "yellow"},
]
