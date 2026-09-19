from __future__ import annotations

import asyncio
from typing import Any, Callable

import cv2
import numpy as np

from ..types import Classification, Frame, ObjectObservation

CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.4}
UNSEEN = "unseen"  # not visible in the scan picture yet: look again once the items in front are gone
SAME_SPOT_MM = 25.0  # an item this close to where one was read last look is the same, unmoved item


class LocalLabelClassifier:
    """Reads a can's label with the team's local reader (llm/classify_image.py): free, no API key.

    That reader sends the crop, plus the photos in reference_pics/, to a vision model served
    by Ollama (OLLAMA_URL, default this laptop; VISION_MODEL) and answers with one of the
    categories in llm/categories.py. Those categories are the piles. Its "not_supported"
    (not a drink we sort) becomes "unknown", which the sort policy sends to reject.

    view: survey  each item's crop is cut from the survey picture (the one positions come from).
          scan    a second picture from the `scan` pose (poses.yaml), closer to the cans so the
                  labels have more pixels. Positions - and so the grasp - still come from the
                  survey, the pose the grasp calibration was measured at; each item's 3D box is
                  projected into the scan picture to cut its crop (perception/scan.py).
    """

    remember = True  # ~10 s per can: run_sort reuses the last read for a can that has not moved

    def __init__(self, cfg: dict[str, Any], read: Callable[[bytes], Any] | None = None):
        if read is None:
            from dotenv import load_dotenv

            load_dotenv()  # the reader takes OLLAMA_URL / VISION_MODEL from the environment when it is imported
            from llm.classify_image import classify_image as read  # imported here: tests pass a fake
        self.read = read
        self.min_side = int(cfg.get("min_crop_side", 640))
        self.pad = float(cfg.get("crop_pad", 0.08))
        self.view: str = cfg.get("view", "survey")
        self.last_views: list = []  # scan view: kept for the debug picture
        self._known: list[tuple[np.ndarray, Classification]] = []  # scan view: what each spot held last look

    @property
    def needs_scan(self) -> bool:
        return self.view == "scan"

    def _crop(self, obs: ObjectObservation, frame: Frame | None):
        """The item's box grown by `pad` per side, as scan_drinks.py does: YOLO's boxes are tight,
        and the edge of a label often holds the brand name."""
        if frame is None or not obs.bbox[2]:
            return obs.crop
        x, y, w, h = obs.bbox
        px, py = int(w * self.pad), int(h * self.pad)
        return frame.color[max(y - py, 0) : y + h + py, max(x - px, 0) : x + w + px]

    def _jpeg(self, crop) -> bytes:
        # A can is ~110x140 px in the survey picture. Enlarging adds no detail, but vision
        # models resize their input, and a tiny image loses the little lettering there is.
        scale = self.min_side / max(1, min(crop.shape[:2]))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

    async def _read(self, crop) -> Classification:
        answer = await asyncio.to_thread(self.read, self._jpeg(crop))  # a blocking HTTP call
        label = "unknown" if answer.label == "not_supported" else answer.label
        meta = {"visible_text": answer.visible_text, "reference": getattr(answer, "closest_reference", "")}
        return Classification(label, CONFIDENCE.get(answer.confidence, 0.4), meta)

    async def classify(self, obs: ObjectObservation, frame: Frame) -> Classification:
        if obs.crop is None or obs.crop.size == 0:
            return Classification("unknown", 0.0, {"why": "no picture of the item"})
        return await self._read(self._crop(obs, frame))

    async def classify_scan(
        self, objects: list[ObjectObservation], scan: Frame, survey: Frame, workspace: dict[str, Any]
    ) -> list[Classification]:
        """view: scan - read each item from its crop of the scan picture. An item hidden behind a
        nearer one there is deferred to a later look; one that has not moved keeps its last read."""
        from ..perception.scan import views

        seen = self.last_views = views(objects, scan, survey, workspace)
        results = []
        for o, v in zip(objects, seen):
            before = next((r for c, r in self._known if np.linalg.norm(c - o.centroid) < SAME_SPOT_MM), None)
            if before is not None:
                results.append(before)
            elif v.readable:
                results.append(await self._read(v.crop))
            else:
                results.append(Classification(UNSEEN, 0.0, {"defer": True, "hidden": v.hidden}))
        self._known = [(o.centroid, r) for o, r in zip(objects, results) if not r.meta.get("defer")]
        return results
