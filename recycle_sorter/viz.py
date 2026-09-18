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


def _color(key: str, i: int) -> tuple[int, int, int]:
    return NAMED.get(key, PALETTE[i % len(PALETTE)])


def draw_layout(
    workspace: dict[str, Any],
    layout: PileLayout,
    objects: list[ObjectObservation] | None = None,
    results: list[Classification] | None = None,
) -> np.ndarray:
    """Top-down map: bounds, unsorted zone, sorted areas, the piles created, and what was seen.

    workspace.map_view picks the viewpoint, so the map matches how you see the table:
      front   you face the arm: arm at the top, +y on your right
      behind  you stand behind the arm: arm at the bottom, +y on your left
    """
    front = workspace.get("map_view", "front") == "front"
    img = np.full((X_RANGE[1] - X_RANGE[0], Y_RANGE[1] - Y_RANGE[0], 3), 245, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def px(x: float, y: float) -> tuple[int, int]:
        if front:
            return int(round(y - Y_RANGE[0])), int(round(x - X_RANGE[0]))
        return int(round(Y_RANGE[1] - y)), int(round(X_RANGE[1] - x))

    def box(x0, x1, y0, y1) -> tuple[tuple[int, int], tuple[int, int]]:
        (ax, ay), (bx, by) = px(x0, y0), px(x1, y1)
        return (min(ax, bx), min(ay, by)), (max(ax, bx), max(ay, by))

    def text(label: str, at: tuple[int, int], color) -> None:
        cv2.putText(img, label, at, font, 0.45, color, 1, cv2.LINE_AA)

    b = workspace["bounds"]
    cv2.rectangle(img, *box(b["x"][0], b["x"][1], b["y"][0], b["y"][1]), (200, 200, 200), 1)
    # The arm sits at the table's -y edge. This marks the configured limit on that side
    # (bounds.y[0]), not a measured table edge.
    e0, e1 = px(X_RANGE[0] + 40, b["y"][0]), px(X_RANGE[1] - 20, b["y"][0])
    cv2.line(img, e0, e1, (90, 90, 90), 3)
    text("edge-side limit", (min(e0[0], e1[0]) + (8 if not front else -112), max(e0[1], e1[1]) - 8), (90, 90, 90))
    for name, a in layout.areas.items():
        tl, br = box(a["x"][0], a["x"][1], a["y"][0], a["y"][1])
        cv2.rectangle(img, tl, br, (225, 225, 225), -1)
        text(f"sorted area: {name}", (tl[0], tl[1] - 8), (130, 130, 130))

    u = workspace["unsorted_zone"]
    tl, br = box(u["x"][0], u["x"][1], u["y"][0], u["y"][1])
    cv2.rectangle(img, tl, br, (255, 0, 255), 2)
    text("unsorted zone", (tl[0], tl[1] - 8), (255, 0, 255))

    for i, (key, zones) in enumerate(layout.zones.items()):
        color = _color(key, i)
        for z in zones:
            tl, br = box(*z.rect)
            cv2.rectangle(img, tl, br, color, 2)
            for n, (sx, sy) in enumerate(z.slots):
                cv2.circle(img, px(sx, sy), 9, color, -1 if n < z.used else 1, cv2.LINE_AA)
            text(f"{key} {z.used}/{len(z.slots)}", (tl[0] + 4, br[1] - 6), (60, 60, 60))

    for i, o in enumerate(objects or []):
        label = results[i].label if results else ""
        r = max(6, int(o.width / 2))
        cv2.circle(img, px(*o.centroid), r, _color(label, i), -1, cv2.LINE_AA)
        cv2.circle(img, px(*o.centroid), r, (60, 60, 60), 1, cv2.LINE_AA)

    ax, ay = px(0, 0)
    cv2.circle(img, (ax, ay), 28, (90, 90, 90), -1, cv2.LINE_AA)
    text("arm", (ax - 14, ay + 5), (255, 255, 255))
    return img
