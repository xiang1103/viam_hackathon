from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .policy.piles import PileLayout
from .types import Classification, ObjectObservation

# BGR. Piles named after a color are drawn in it; anything else cycles the palette.
NAMED = {
    "red": (60, 60, 220), "orange": (40, 140, 240), "yellow": (60, 215, 235), "green": (80, 175, 70),
    "blue": (215, 110, 50), "purple": (170, 80, 150), "black": (40, 40, 40), "grey": (140, 140, 140),
    "white": (235, 235, 235), "reject": (110, 110, 150),
}
PALETTE = [(180, 120, 40), (70, 160, 200), (150, 90, 170), (90, 170, 120), (60, 110, 200), (160, 160, 70)]

X_RANGE, Y_RANGE = (-40, 760), (-520, 520)  # mm of table shown; 1 px = 1 mm


def _px(x: float, y: float) -> tuple[int, int]:
    """World mm -> pixel, viewed from behind the arm: +x (forward) is up, +y is left."""
    return int(round(Y_RANGE[1] - y)), int(round(X_RANGE[1] - x))


def _rect(img, x0, x1, y0, y1, color, thickness) -> None:
    cv2.rectangle(img, _px(x1, y1), _px(x0, y0), color, thickness)


def _color(key: str, i: int) -> tuple[int, int, int]:
    return NAMED.get(key, PALETTE[i % len(PALETTE)])


def draw_layout(
    workspace: dict[str, Any],
    layout: PileLayout,
    objects: list[ObjectObservation] | None = None,
    results: list[Classification] | None = None,
) -> np.ndarray:
    """Top-down map: bounds, unsorted zone, sorted areas, the piles created, and what was seen."""
    img = np.full((X_RANGE[1] - X_RANGE[0], Y_RANGE[1] - Y_RANGE[0], 3), 245, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    b = workspace["bounds"]
    _rect(img, b["x"][0], b["x"][1], b["y"][0], b["y"][1], (200, 200, 200), 1)
    for name, a in layout.areas.items():
        _rect(img, a["x"][0], a["x"][1], a["y"][0], a["y"][1], (225, 225, 225), -1)
        cv2.putText(img, f"sorted area: {name}", _px(a["x"][1] + 8, a["y"][1]), font, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

    u = workspace["unsorted_zone"]
    _rect(img, u["x"][0], u["x"][1], u["y"][0], u["y"][1], (255, 0, 255), 2)
    cv2.putText(img, "unsorted zone", _px(u["x"][1] + 8, u["y"][1]), font, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

    for i, (key, zones) in enumerate(layout.zones.items()):
        color = _color(key, i)
        for z in zones:
            x0, x1, y0, y1 = z.rect
            _rect(img, x0, x1, y0, y1, color, 2)
            for n, (sx, sy) in enumerate(z.slots):
                cv2.circle(img, _px(sx, sy), 9, color, -1 if n < z.used else 1, cv2.LINE_AA)
            cv2.putText(img, f"{key} {z.used}/{len(z.slots)}", _px(x0 - 6, y1), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)

    for i, o in enumerate(objects or []):
        label = results[i].label if results else ""
        cx, cy = _px(*o.centroid)
        cv2.circle(img, (cx, cy), max(6, int(o.width / 2)), _color(label, i), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), max(6, int(o.width / 2)), (60, 60, 60), 1, cv2.LINE_AA)

    cv2.circle(img, _px(0, 0), 28, (90, 90, 90), -1, cv2.LINE_AA)
    cv2.putText(img, "arm", (_px(0, 0)[0] - 14, _px(0, 0)[1] + 5), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img
