from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import ROOT
from ..types import Frame, ObjectObservation
from .segment import finish, in_unsorted_zone, level, observe
from .silhouette import fit_upright, is_upright

log = logging.getLogger(__name__)


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    label: str
    confidence: float
    mask: np.ndarray | None = None  # HxW bool at image resolution, when the model segments


class YoloDetector:
    """2D detections from an Ultralytics model. The model is loaded on first use.

    kind: yoloe | world  open-vocabulary: `classes` are plain words, no training needed
          yolo           a fixed-class model (COCO-pretrained or your own .pt); `classes`
                         then FILTERS its class names, or leave empty to keep them all
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._model = None
        # One load and one prediction at a time: the launch warm-up runs in a thread and the first
        # order may ask for boxes before it is done - it then waits instead of loading a second copy.
        self._lock = threading.RLock()

    @property
    def model(self):
        with self._lock:
            return self._load()

    def warm_up(self) -> None:
        """Load the model and run it once (the first prediction is ~1 s slower than later ones)."""
        self.detect(np.full((64, 64, 3), 200, np.uint8))

    def _load(self):
        if self._model is None:
            try:
                import ultralytics
            except ImportError as e:
                raise SystemExit("perception: yolo needs ultralytics - run: .venv/bin/pip install ultralytics") from e
            kind, classes = self.cfg.get("kind", "yoloe"), list(self.cfg.get("classes") or [])
            weights = (ROOT / self.cfg["weights"]).resolve()
            weights.parent.mkdir(parents=True, exist_ok=True)
            if kind == "yoloe":
                import torch

                # Turning words into prompts needs a text encoder; the result is tiny, so it is
                # kept and the encoder skipped on every later start.
                cache = weights.parent / f"{weights.stem}-{'-'.join(classes)}.prompts.pt"
                if weights.exists() and cache.exists():
                    # Everything is on disk: no chdir, which would change the working directory
                    # of the whole process while this may be loading in a background thread.
                    model = ultralytics.YOLOE(str(weights))
                    model.set_classes(classes, torch.load(cache))
                else:
                    # Ultralytics downloads into the current directory: work from the weights folder
                    # so the model (and the 600 MB text encoder) land there once.
                    here = os.getcwd()
                    os.chdir(weights.parent)
                    try:
                        model = ultralytics.YOLOE(weights.name)
                        if cache.exists():
                            prompts = torch.load(cache)
                        else:
                            prompts = model.get_text_pe(classes)
                            torch.save(prompts, cache)
                        model.set_classes(classes, prompts)
                    finally:
                        os.chdir(here)
            elif kind == "world":
                model = ultralytics.YOLOWorld(str(weights))
                model.set_classes(classes)
            else:
                model = ultralytics.YOLO(str(weights))
            self._model = model
            log.info("loaded %s model %s", kind, weights)
        return self._model

    def detect(self, image_bgr: np.ndarray) -> list[Box]:
        """Blocking (~0.25 s, several seconds on first use): call it from a thread in async code."""
        with self._lock:
            return self._detect(image_bgr)

    def _detect(self, image_bgr: np.ndarray) -> list[Box]:
        cfg = self.cfg
        result = self.model.predict(
            image_bgr, conf=cfg.get("confidence", 0.25), iou=cfg.get("iou", 0.5), imgsz=cfg.get("imgsz", 1280), verbose=False
        )[0]
        keep = set(cfg.get("classes") or []) if cfg.get("kind") == "yolo" else None
        h, w = image_bgr.shape[:2]
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        boxes = []
        for i, (xyxy, conf, cls) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())):
            label = result.names[int(cls)]
            if keep and label not in keep:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in xyxy)
            mask = None
            if masks is not None:
                mask = cv2.resize(masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            boxes.append(Box(max(x0, 0), max(y0, 0), min(x1, w), min(y1, h), label, float(conf), mask))
        return boxes


_shared: dict[str, YoloDetector] = {}
_shared_lock = threading.Lock()


def shared_detector(cfg: dict[str, Any]) -> YoloDetector:
    """The one YoloDetector for this config, for the whole process. The survey picture, the scan
    picture and the launch warm-up all use it, so the model is loaded once."""
    key = json.dumps(cfg, sort_keys=True)
    with _shared_lock:
        if key not in _shared:
            _shared[key] = YoloDetector(cfg)
        return _shared[key]


def _own_top(blob: np.ndarray, height: np.ndarray, top_band: float) -> np.ndarray:
    """Drop a neighbour's top from a blob, so the footprint is this item's alone.

    Seen from a tilted camera, the lid of the can behind touches this can's lid in the
    picture, and the blob inside the box then holds both: a "can" 124 mm long. Tops are
    separate surfaces joined only by a thin neck of pixels, so eroding splits them; the
    largest piece is the item the box was drawn around.
    """
    top = blob & (height > np.percentile(height[blob], 95) - top_band)
    kernel = np.ones((5, 5), np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.erode(top.astype(np.uint8), kernel), connectivity=8)
    if n < 2:
        return blob
    own = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # Everything at top height that is not part of (or right next to) the item's own top goes:
    # the neighbour's lid, and the few stray pixels of it that erosion wiped out altogether.
    near_own = cv2.dilate((labels == own).astype(np.uint8), kernel, iterations=2).astype(bool)
    return blob & ~(top & ~near_own)


def _centre_on_silhouette(o: ObjectObservation, b: Box, frame: Frame, workspace: dict[str, Any]) -> None:
    """A standing can's position from its outline in the colour picture (perception/silhouette.py)
    instead of from its depth points, which drop out on a shiny lid. The depth answer is kept when
    there is no mask, the outline fits the mask badly (a can lying down, two cans in one mask), or
    the fit wandered off."""
    cfg = workspace.get("silhouette") or {}
    if not cfg.get("enabled", False) or b.mask is None or not is_upright(o, workspace):
        return
    fit = fit_upright(frame, b.mask, o.centroid, o.top_z, workspace)
    if fit is None:
        return
    xy, _, iou = fit
    moved = float(np.linalg.norm(xy - o.centroid))
    if iou < cfg.get("min_iou", 0.75) or moved > cfg.get("max_shift", 35.0):
        log.info("item at (%.0f, %.0f): silhouette fit not used (IoU %.2f, %.0f mm away)", *o.centroid, iou, moved)
        return
    o.centroid, o.grasp_xy = xy.copy(), xy.copy()


def observations_from_boxes(
    boxes: list[Box], frame: Frame, workspace: dict[str, Any], rejected: list[tuple[Box, str]] | None = None
) -> list[ObjectObservation]:
    """Turn 2D detections into the 3D objects the rest of the pipeline works with.

    YOLO says WHICH pixels are an item; the aligned depth image says where those pixels are
    in the world. Inside each box only what stands above the table is kept, and of that the
    one connected blob nearest the box's middle - so table, and a neighbour poking into the
    corner of the box, are left out. Size, top height and grasp then come from observe(),
    exactly as for depth-only detection.

    `rejected`, when given, receives (box, reason) for every box that did not become an item.
    """
    rejected = [] if rejected is None else rejected
    seg, table_top = workspace["segmentation"], workspace["table_top"]
    pts, valid, _ = level(frame, workspace)
    height = pts[..., 2] - table_top
    standing = valid & (height > seg["min_height"]) & (height < seg["max_object_dim"])

    found = []
    for b in boxes:
        region = np.zeros(standing.shape, bool)
        region[b.y0 : b.y1, b.x0 : b.x1] = True
        if b.mask is not None:
            region &= b.mask
        candidate = (standing & region).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        if n < 2:
            rejected.append((b, "nothing above the table"))  # a false detection, or no depth there
            continue
        middle = np.array([(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2])
        big = [i for i in range(1, n) if stats[i][4] >= seg["min_area_px"]] or list(range(1, n))
        best = min(big, key=lambda i: np.linalg.norm(centroids[i] - middle))
        m = _own_top(labels == best, height, seg["top_band"])
        o = observe(pts[m], frame, workspace, mask=m, source_label=b.label)
        if o is None:
            rejected.append((b, "too few depth points"))
            continue
        if o.height < seg.get("min_item_height", 0):
            rejected.append((b, "too low for a can"))  # a cable, a charger: what a loose prompt also boxes
            continue
        # The label is on the can's side, which the depth blob may only partly cover: keep
        # YOLO's whole box as the picture of the item.
        o.bbox = (b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0)
        o.crop = frame.color[b.y0 : b.y1, b.x0 : b.x1].copy()
        o.detection_confidence = b.confidence
        _centre_on_silhouette(o, b, frame, workspace)
        found.append((b, o))
    kept = finish([o for _, o in found], workspace)
    for b, o in found:
        if o not in kept:
            outside = not in_unsorted_zone(o, workspace)
            rejected.append((b, "outside the unsorted zone" if outside else "too big for one item"))
    return kept
