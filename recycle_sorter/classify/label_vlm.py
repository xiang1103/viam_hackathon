from __future__ import annotations

import asyncio
from typing import Any, Callable

import cv2

from ..types import Classification, Frame, ObjectObservation

CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.4}


class LocalLabelClassifier:
    """Reads a can's label with the team's local reader (llm/classify_image.py): free, no API key.

    That reader sends the crop, plus the photos in reference_pics/, to a vision model served
    by Ollama (OLLAMA_URL, default this laptop; VISION_MODEL) and answers with one of the
    categories in llm/categories.py. Those categories are the piles. Its "not_supported"
    (not a drink we sort) becomes "unknown", which the sort policy sends to reject.
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

    async def classify(self, obs: ObjectObservation, frame: Frame) -> Classification:
        if obs.crop is None or obs.crop.size == 0:
            return Classification("unknown", 0.0, {"why": "no picture of the item"})
        answer = await asyncio.to_thread(self.read, self._jpeg(self._crop(obs, frame)))  # a blocking HTTP call
        label = "unknown" if answer.label == "not_supported" else answer.label
        meta = {"visible_text": answer.visible_text, "reference": getattr(answer, "closest_reference", "")}
        return Classification(label, CONFIDENCE.get(answer.confidence, 0.4), meta)
