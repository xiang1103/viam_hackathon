from __future__ import annotations

import asyncio
from typing import Any, Callable

import cv2
import numpy as np

from ..types import Classification, Frame, ObjectObservation

CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.4}
SAME_SPOT_MM = 25.0  # an item this close to where one was read last look is the same, unmoved item
VIEW_NAMES = {"survey": "from above, farther away", "scan": "closer, from the side at an angle"}


class LocalLabelClassifier:
    """Reads a can's label with the team's local reader (llm/classify_image.py): free, no API key.

    That reader sends the crop, plus the photos in reference_pics/, to a vision model served
    by Ollama (OLLAMA_URL, default this laptop; VISION_MODEL) and answers with one of the
    categories in llm/categories.py. Those categories are the piles. Its "not_supported"
    (not a drink we sort) becomes "unknown", which the sort policy sends to reject.

    view: survey  each item's crop is cut from the survey picture (the one positions come from).
          scan    TWO crops per item go to the reader together: its box in the survey picture (from
                  above) and its box in a second, closer picture from the `scan` pose (poses.yaml).
                  Positions - and so the grasp - still come from the survey, the pose the grasp
                  calibration was measured at. In the scan picture, each item's 3D box is projected
                  (perception/scan.py) and linked one-to-one to the nearest YOLO box there, so the
                  crop is tight around one can: the projection alone lands tens of pixels off.
                  Every box in either picture is read: no box in the scan picture -> the survey
                  crop alone; a scan box no item links to -> that scan crop alone.
    """

    remember = True  # ~10 s per can: run_sort reuses the last read for a can that has not moved

    def __init__(
        self,
        cfg: dict[str, Any],
        read: Callable[..., Any] | None = None,
        detect: Callable[[np.ndarray], list] | None = None,
    ):
        if read is None:
            from dotenv import load_dotenv

            load_dotenv()  # the reader takes OLLAMA_URL / VISION_MODEL from the environment when it is imported
            from llm.classify_image import classify_image as read  # imported here: tests pass a fake
        self.read = read  # read(jpeg) or read([jpeg, jpeg], view_names) -> an llm.classify_image answer
        self._detect = detect  # scan view: YOLO on the scan picture; loaded on first use, tests pass a fake
        self.min_side = int(cfg.get("min_crop_side", 640))
        self.pad = float(cfg.get("crop_pad", 0.08))
        self.view: str = cfg.get("view", "survey")
        self.last_views: list = []  # scan view: kept for the debug picture
        self.last_scan_only: list[Classification] = []  # scan view: cans YOLO found only in the scan picture
        self.last_links: list[dict] = []  # scan view: which survey box and scan box each read came from
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

    async def _read(self, *crops, views: tuple[str, ...] = ()) -> Classification:
        """One answer from one crop, or from several crops of the same item (named in `views`)."""
        if len(crops) == 1:
            answer = await asyncio.to_thread(self.read, self._jpeg(crops[0]))  # a blocking HTTP call
        else:
            answer = await asyncio.to_thread(self.read, [self._jpeg(c) for c in crops], list(views))
        label = "unknown" if answer.label == "not_supported" else answer.label
        meta = {"visible_text": answer.visible_text, "reference": getattr(answer, "closest_reference", "")}
        return Classification(label, CONFIDENCE.get(answer.confidence, 0.4), meta)

    async def classify(self, obs: ObjectObservation, frame: Frame) -> Classification:
        if obs.crop is None or obs.crop.size == 0:
            return Classification("unknown", 0.0, {"why": "no picture of the item"})
        return await self._read(self._crop(obs, frame))

    def detect(self, image: np.ndarray) -> list:
        if self._detect is None:
            from ..config import load_yaml
            from ..perception.yolo import YoloDetector

            self._detect = YoloDetector(load_yaml("machine.yaml")["yolo"]).detect
        return self._detect(image)

    def _box_crop(self, image: np.ndarray, b) -> np.ndarray:
        w, h = b.x1 - b.x0, b.y1 - b.y0
        px, py = int(w * self.pad), int(h * self.pad)
        return image[max(b.y0 - py, 0) : b.y1 + py, max(b.x0 - px, 0) : b.x1 + px].copy()

    def _link(self, seen: list, scan: Frame, max_hidden: float) -> tuple[list, list[int], list]:
        """Link each item from the survey to at most one YOLO box in the scan picture, and back.

        The item's projected 3D box only says roughly where it is (the calibration is off by tens
        of pixels at the scan pose); a YOLO box is tight around exactly one can, so the nearest
        YOLO box replaces the projection. Pairing is one-to-one, closest first, and items that are
        not hidden behind a nearer one choose first - so an item hidden behind another does not
        take the front item's box. Returns (scan view per item, YOLO box index per item or -1,
        every YOLO box). Boxes no item links to are the cans seen only in the scan picture."""
        from ..perception.scan import ScanView

        boxes = self.detect(scan.color)
        centre = lambda x, y, w, h: np.array([x + w / 2, y + h / 2])  # noqa: E731
        pairs = []
        for i, v in enumerate(seen):
            if v.box is None:
                continue
            reach = max(v.box[2], v.box[3])  # the projected box already carries a generous margin
            for j, b in enumerate(boxes):
                d = float(np.linalg.norm(centre(*v.box) - centre(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0)))
                if d < reach:
                    pairs.append((v.hidden > max_hidden, d, i, j))
        linked: list = [ScanView(None, v.distance, v.hidden) for v in seen]
        box_of = [-1] * len(seen)
        used_boxes: set[int] = set()
        for _, _, i, j in sorted(pairs):
            if box_of[i] >= 0 or j in used_boxes:
                continue
            box_of[i] = j
            used_boxes.add(j)
            b = boxes[j]
            linked[i] = ScanView((b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0), seen[i].distance, seen[i].hidden,
                                 self._box_crop(scan.color, b))
        return linked, box_of, boxes

    async def classify_scan(
        self, objects: list[ObjectObservation], scan: Frame, survey: Frame, workspace: dict[str, Any]
    ) -> list[Classification]:
        """view: scan - every can with a YOLO box in EITHER picture is read.

        An item with a box in both pictures is read from both crops in one request. One that has a
        box only in the survey picture is read from its survey crop alone. A box only in the scan
        picture (no item from the survey links to it) is read from its scan crop alone; it has no
        position, so it is reported in `last_scan_only`, not returned. One that has not moved since
        the last look keeps its last read. Each result's meta["link"] says which boxes it came from;
        `last_links` lists them all."""
        from ..perception.scan import views

        projected = views(objects, scan, survey, workspace)
        seen, box_of, boxes = self._link(projected, scan, workspace["scan"]["max_hidden"])
        self.last_views = seen
        results = []
        for i, (o, v) in enumerate(zip(objects, seen)):
            link = {
                "can": i,
                "survey_box": list(o.bbox) if o.bbox[2] else None,
                "scan_box": list(v.box) if v.readable else None,
                "scan_yolo_index": box_of[i] if box_of[i] >= 0 else None,
            }
            before = next((r for c, r in self._known if np.linalg.norm(c - o.centroid) < SAME_SPOT_MM), None)
            top = self._crop(o, survey) if o.crop is not None and o.crop.size else None
            if before is not None:
                link["views"], r = before.meta.get("link", {}).get("views", []), before
                results.append(Classification(r.label, r.confidence, {**r.meta, "link": link, "remembered": True}))
                continue
            if top is not None and v.readable:
                link["views"] = ["survey", "scan"]
                r = await self._read(top, v.crop, views=(VIEW_NAMES["survey"], VIEW_NAMES["scan"]))
            elif top is not None:
                link["views"] = ["survey"]
                r = await self._read(top)
            elif v.readable:
                link["views"] = ["scan"]
                r = await self._read(v.crop)
            else:
                link["views"] = []
                r = Classification("unknown", 0.0, {"why": "no picture of the item"})
            r.meta["link"] = link
            results.append(r)
        self._known = [(o.centroid, r) for o, r in zip(objects, results)]

        # Cans YOLO found only in the scan picture: read too, but they have no position to pick from.
        linked_boxes = {j for j in box_of if j >= 0}
        self.last_scan_only = []
        for n, (j, b) in enumerate((j, b) for j, b in enumerate(boxes) if j not in linked_boxes):
            r = await self._read(self._box_crop(scan.color, b))
            r.meta["link"] = {"can": f"scan-{n}", "survey_box": None,
                              "scan_box": [b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0],
                              "scan_yolo_index": j, "views": ["scan"]}
            self.last_scan_only.append(r)
        self.last_links = [{**r.meta["link"], "label": r.label, "confidence": r.confidence}
                           for r in results + self.last_scan_only]
        return results
