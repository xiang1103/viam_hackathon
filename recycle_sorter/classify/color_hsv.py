from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..types import Classification, Frame, ObjectObservation


class HSVColorClassifier:
    """Dominant color of the pixels inside the object's mask, matched to the nearest named hue."""

    def __init__(self, colors: dict[str, Any]):
        self.hues: dict[str, float] = colors["hues"]
        self.min_sat = colors["min_saturation"]
        self.min_val = colors["min_value"]

    async def classify(self, obs: ObjectObservation, frame: Frame) -> Classification:
        # Erode so edge pixels (shadow, table bleed) don't vote.
        inner = cv2.erode(obs.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if inner.sum() < 50:
            inner = obs.mask
        hsv = cv2.cvtColor(frame.color, cv2.COLOR_BGR2HSV)[inner]
        h, s, v = hsv[:, 0].astype(float) * 2.0, hsv[:, 1], hsv[:, 2]  # OpenCV hue is 0-179

        chromatic = (s >= self.min_sat) & (v >= self.min_val)
        if chromatic.mean() < 0.5:
            brightness = float(np.median(v))
            label = "black" if brightness < 70 else "white" if brightness > 180 else "grey"
            return Classification(label, 1.0 - float(chromatic.mean()), {"value": brightness})

        # Hue is circular (red wraps at 0/360), so average on the unit circle.
        rad = np.deg2rad(h[chromatic])
        hue = float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)

        def dist(a: float, b: float) -> float:
            return min(abs(a - b), 360.0 - abs(a - b))

        ranked = sorted(self.hues, key=lambda name: dist(hue, self.hues[name]))
        best = dist(hue, self.hues[ranked[0]])
        runner_up = dist(hue, self.hues[ranked[1]]) if len(ranked) > 1 else 180.0
        confidence = 1.0 - best / max(best + runner_up, 1e-6)
        return Classification(ranked[0], float(confidence), {"hue": hue})
