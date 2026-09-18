from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import ROOT
from ..types import Frame, ObjectObservation
from .segment import finish, level, observe

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

    @property
    def model(self):
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

                # Ultralytics downloads into the current directory: work from the weights folder so
                # the model (and the 600 MB text encoder) land there once instead of wherever we run.
                here = os.getcwd()
                os.chdir(weights.parent)
                try:
                    model = ultralytics.YOLOE(weights.name)
                    # Turning words into prompts needs that text encoder; the result is tiny, so
                    # keep it and skip the encoder on every later start.
                    cache = weights.parent / f"{weights.stem}-{'-'.join(classes)}.prompts.pt"
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


def observations_from_boxes(boxes: list[Box], frame: Frame, workspace: dict[str, Any]) -> list[ObjectObservation]:
    """Turn 2D detections into the 3D objects the rest of the pipeline works with.

    YOLO says WHICH pixels are an item; the aligned depth image says where those pixels are
    in the world. Inside each box only what stands above the table is kept, and of that the
    one connected blob nearest the box's middle - so table, and a neighbour poking into the
    corner of the box, are left out. Size, top height and grasp then come from observe(),
    exactly as for depth-only detection.
    """
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
            continue  # nothing above the table inside this box: a false detection, or no depth there
        middle = np.array([(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2])
        big = [i for i in range(1, n) if stats[i][4] >= seg["min_area_px"]] or list(range(1, n))
        best = min(big, key=lambda i: np.linalg.norm(centroids[i] - middle))
        m = _own_top(labels == best, height, seg["top_band"])
        o = observe(pts[m], frame, workspace, mask=m, source_label=b.label)
        if o is None:
            continue
        # The label is on the can's side, which the depth blob may only partly cover: keep
        # YOLO's whole box as the picture of the item.
        o.bbox = (b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0)
        o.crop = frame.color[b.y0 : b.y1, b.x0 : b.x1].copy()
        o.detection_confidence = b.confidence
        found.append(o)
    return finish(found, workspace)
