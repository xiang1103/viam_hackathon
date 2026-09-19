from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from ..types import Classification, Frame, ObjectObservation

CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.4}
SAME_SPOT_MM = 25.0  # an item this close to where one was read last look is the same, unmoved item
VIEW_NAMES = {"survey": "from above, farther away", "scan": "closer, from the side at an angle"}
SAME_LOOK = 0.85  # how alike (0-1) two crops' colours must be for "the same can": see signature()
SURE = 0.9  # a read this confident settles one item of an order (the reader's "high")


@dataclass
class Remembered:
    """What stood at a spot the last time it was read, and what it looked like."""

    xy: np.ndarray
    look: np.ndarray  # signature() of the crop that was read
    result: Classification


# One memory for the whole process. run_sort builds a new classifier for every order, so a memory kept
# on the classifier was gone by the next order and every can on the table was read again (~10 s each).
SESSION_MEMORY: list[Remembered] = []


def signature(crop: np.ndarray) -> np.ndarray:
    """A colour fingerprint of the middle of a crop: hue x saturation, plus brightness, as one histogram.

    A remembered label is only reused when the can at that spot still LOOKS the same: between orders
    someone may have swapped it. Measured 2026-09-19 on saved frames with likeness() below: two pictures
    of the same unmoved can score 0.92 (median), different cans 0.59; at SAME_LOOK = 0.85 a different can
    passes in under 1 % of pairs (and it must also stand within SAME_SPOT_MM of the old one), while ~14 %
    of unmoved cans are read again for nothing - which only costs the ~10 s it would have cost anyway."""
    h, w = crop.shape[:2]
    mid = crop[int(h * 0.2) : max(int(h * 0.8), int(h * 0.2) + 1), int(w * 0.2) : max(int(w * 0.8), int(w * 0.2) + 1)]
    hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([
        cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel(),
        cv2.calcHist([hsv], [2], None, [6], [0, 256]).ravel(),
    ]).astype(np.float32)
    return hist / max(float(hist.sum()), 1.0)


def likeness(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection of two signatures: 1 = identical colours, 0 = nothing in common."""
    return float(np.minimum(a, b).sum())


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
        memory: list[Remembered] | None = None,
    ):
        if read is None:
            from dotenv import load_dotenv

            load_dotenv()  # the reader takes OLLAMA_URL / VISION_MODEL from the environment when it is imported
            from llm.classify_image import classify_image as read  # imported here: tests pass a fake
        self.read = read  # read(jpeg) or read([jpeg, jpeg], view_names) -> an llm.classify_image answer
        self._detect = detect  # scan view: YOLO on the scan picture (the shared model); tests pass a fake
        self.min_side = int(cfg.get("min_crop_side", 640))
        self.pad = float(cfg.get("crop_pad", 0.08))
        self.view: str = cfg.get("view", "survey")
        self.survey_crop = bool(cfg.get("survey_crop", True))  # scan view: also send the survey crop with the scan crop
        self.last_views: list = []  # scan view: kept for the debug picture
        self.last_scan_only: list[Classification] = []  # scan view: cans YOLO found only in the scan picture
        self.last_links: list[dict] = []  # scan view: which survey box and scan box each read came from
        # scan view: what each spot held when it was last read. app.make_classifier passes SESSION_MEMORY, so
        # it outlives this classifier; on its own (tests) a classifier starts with an empty one.
        self.memory: list[Remembered] = [] if memory is None else memory
        # scan view, set by run_sort for an ORDER: {label: how many are still to fetch}. Reading stops as soon
        # as sure reads cover it; None = read everything (a sort needs every label).
        self.wanted: dict[str, int] | None = None

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
            from ..perception.yolo import shared_detector

            # The same model as the survey picture's (loaded once per process, warmed at launch).
            self._detect = shared_detector(load_yaml("machine.yaml")["yolo"]).detect
        return self._detect(image)

    def _box_crop(self, image: np.ndarray, b) -> np.ndarray:
        w, h = b.x1 - b.x0, b.y1 - b.y0
        px, py = int(w * self.pad), int(h * self.pad)
        return image[max(b.y0 - py, 0) : b.y1 + py, max(b.x0 - px, 0) : b.x1 + px].copy()

    def _link(self, seen: list, scan: Frame, boxes: list, max_hidden: float) -> tuple[list, list[int], list]:
        """Link each item from the survey to at most one YOLO box in the scan picture, and back.

        The item's projected 3D box only says roughly where it is (the calibration is off by tens
        of pixels at the scan pose); a YOLO box is tight around exactly one can, so the nearest
        YOLO box replaces the projection. Pairing is one-to-one, closest first, and items that are
        not hidden behind a nearer one choose first - so an item hidden behind another does not
        take the front item's box. Returns (scan view per item, YOLO box index per item or -1,
        every YOLO box). Boxes no item links to are the cans seen only in the scan picture."""
        from ..perception.scan import ScanView

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
        boxes = await asyncio.to_thread(self.detect, scan.color)  # ~0.25 s: off the event loop
        seen, box_of, boxes = self._link(projected, scan, boxes, workspace["scan"]["max_hidden"])
        self.last_views = seen
        def covered() -> bool:
            """An order, and sure reads already account for every item of it: nothing more needs reading."""
            if not self.wanted:
                return False
            sure = [r.label for r in results if r is not None and r.confidence >= SURE]
            return all(sure.count(label) >= n for label, n in self.wanted.items() if n > 0)

        def link_of(i: int) -> dict:
            o, v = objects[i], seen[i]
            return {"can": i, "survey_box": list(o.bbox) if o.bbox[2] else None,
                    "scan_box": list(v.box) if v.readable else None,
                    "scan_yolo_index": box_of[i] if box_of[i] >= 0 else None}

        # What each item looks like now: its scan crop when it has one (the view the label is read from).
        tops = [self._crop(o, survey) if o.crop is not None and o.crop.size else None for o in objects]
        looks = [signature(v.crop if v.readable else t) if (v.readable or t is not None) else None
                 for v, t in zip(seen, tops)]

        # 1. Remembered first - they cost nothing. The spot must match AND the can must still look the same.
        results: list[Classification | None] = [None] * len(objects)
        for i, o in enumerate(objects):
            m = next((m for m in self.memory if np.linalg.norm(m.xy - o.centroid) < SAME_SPOT_MM), None)
            if m is not None and looks[i] is not None and likeness(m.look, looks[i]) >= SAME_LOOK:
                link = {**link_of(i), "views": m.result.meta.get("link", {}).get("views", [])}
                results[i] = Classification(m.result.label, m.result.confidence,
                                            {**m.result.meta, "link": link, "remembered": True})

        # 2. Read the rest, one request each (~10 s) - until the order is covered.
        for i, (o, v) in enumerate(zip(objects, seen)):
            if results[i] is not None:
                continue
            link, top = link_of(i), tops[i]
            if covered():
                link["views"] = []
                results[i] = Classification("unread", 0.0, {"why": "the order was already covered", "link": link})
                continue
            if top is not None and v.readable and not self.survey_crop:
                top = None  # a top-view survey shows the lid, not the label: the scan crop alone is read
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
            results[i] = r

        # Remember what was read (not what was skipped), replacing whatever was known about those spots.
        read = [(o, look, r) for o, look, r in zip(objects, looks, results) if look is not None and r.label != "unread"]
        self.memory[:] = [m for m in self.memory
                          if not any(np.linalg.norm(m.xy - o.centroid) < SAME_SPOT_MM for o in objects)]
        self.memory.extend(Remembered(o.centroid.copy(), look, r) for o, look, r in read)

        # Cans YOLO found only in the scan picture: read too, but they have no position to pick from -
        # so not once an order is covered: they could not be fetched anyway.
        linked_boxes = {j for j in box_of if j >= 0}
        self.last_scan_only = []
        for n, (j, b) in enumerate((j, b) for j, b in enumerate(boxes) if j not in linked_boxes):
            if covered():
                break
            r = await self._read(self._box_crop(scan.color, b))
            r.meta["link"] = {"can": f"scan-{n}", "survey_box": None,
                              "scan_box": [b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0],
                              "scan_yolo_index": j, "views": ["scan"]}
            self.last_scan_only.append(r)
        self.last_links = [{**r.meta["link"], "label": r.label, "confidence": r.confidence}
                           for r in results + self.last_scan_only]
        return results
