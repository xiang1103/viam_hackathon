"""Camera picture -> YOLO boxes -> crop each box -> local VLM names its drink category.

    python scan_drinks.py --camera                 # live picture from the robot (moves nothing)
    python scan_drinks.py path/to/picture.jpg      # any saved picture

Every run is recorded in data/scans/<timestamp>/:
    picture.jpg     the full camera picture
    detected.jpg    the picture with each box, its number and its category drawn on
    crops/NN.jpg    exactly what the VLM saw for box NN
    results.json    per box: bbox, YOLO confidence, crop file, category, VLM confidence, text read

From code:
    from scan_drinks import scan
    items = scan(image_bgr, detector)     # detector = CanDetector(), created once
    for it in items: print(it.bbox, it.label, it.confidence)

YOLO only finds WHERE the cans and bottles are (it calls everything "can" or "bottle").
WHAT each one is comes from llm/classify_image.py, using the categories in llm/categories.py.
Needs `ollama serve` running with qwen2.5vl:7b pulled.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detect_cans import CanDetector
from llm.classify_image import classify_image
from recycle_sorter.config import ROOT

SCANS_DIR = ROOT / "data" / "scans"
CROP_PAD = 0.08  # grow each box by this fraction per side: YOLO boxes are tight, and the
                 # edge of a label often holds the brand name
CROP_MIN_SIDE = 640  # small crops are enlarged so the VLM can read the print on them


@dataclass
class ScannedItem:
    index: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in picture pixels (YOLO's box, unpadded)
    yolo_confidence: float
    label: str  # category from llm/categories.py, or "not_supported"
    confidence: str  # VLM confidence: high | medium | low
    visible_text: str  # what the VLM read on the container
    crop: str = ""  # crop file, relative to the scan folder, when recorded


def crop_box(image_bgr: np.ndarray, box) -> np.ndarray:
    """The box, padded by CROP_PAD and enlarged to CROP_MIN_SIDE if it is small."""
    h, w = image_bgr.shape[:2]
    pad_x, pad_y = int((box.x1 - box.x0) * CROP_PAD), int((box.y1 - box.y0) * CROP_PAD)
    crop = image_bgr[max(box.y0 - pad_y, 0):min(box.y1 + pad_y, h),
                     max(box.x0 - pad_x, 0):min(box.x1 + pad_x, w)]
    scale = CROP_MIN_SIDE / max(crop.shape[:2])
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def scan(image_bgr: np.ndarray, detector: CanDetector, out_dir: Path | None = None,
         log=print) -> list[ScannedItem]:
    """Find every can/bottle, classify each crop, and (with out_dir) record crops + results."""
    start = time.perf_counter()
    boxes = detector(image_bgr)
    log(f"YOLO: {len(boxes)} cans/bottles in {time.perf_counter() - start:.2f} s")
    if out_dir:
        (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    items = []
    for i, box in enumerate(boxes):
        crop = crop_box(image_bgr, box)
        ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            log(f"  {i}: could not encode the crop, skipped")
            continue
        start = time.perf_counter()
        c = classify_image(jpeg.tobytes())
        item = ScannedItem(i, (box.x0, box.y0, box.x1, box.y1), round(box.confidence, 3),
                           c.label, c.confidence, c.visible_text)
        if out_dir:
            item.crop = f"crops/{i:02d}.jpg"
            (out_dir / item.crop).write_bytes(jpeg.tobytes())
        log(f"  {i}: {item.label:<16} ({item.confidence:<6}) yolo {box.confidence:.2f}  "
            f"box {item.bbox}  text={c.visible_text[:40]!r}  [{time.perf_counter() - start:.1f} s]")
        items.append(item)

    if out_dir:
        cv2.imwrite(str(out_dir / "picture.jpg"), image_bgr)
        cv2.imwrite(str(out_dir / "detected.jpg"), draw(image_bgr, items))
        (out_dir / "results.json").write_text(json.dumps([asdict(it) for it in items], indent=2))
    return items


def draw(image_bgr: np.ndarray, items: list[ScannedItem]) -> np.ndarray:
    out = image_bgr.copy()
    for it in items:
        color = (0, 200, 0) if it.confidence == "high" else (0, 200, 255) if it.confidence == "medium" \
            else (0, 0, 255)  # green / orange / red by VLM confidence
        x0, y0, x1, y1 = it.bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, f"{it.index} {it.label}", (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Find cans/bottles with YOLO and name each with the VLM.")
    ap.add_argument("image", nargs="?", type=Path, help="a color picture (png / jpg)")
    ap.add_argument("--camera", action="store_true", help="take a live picture from the robot (read-only)")
    args = ap.parse_args()
    if bool(args.image) == args.camera:
        ap.error("give a picture, or --camera")

    if args.camera:
        import asyncio

        from recycle_sorter.io.recorder import save_frame
        from recycle_sorter.io.robot import LiveRobot

        async def snap():
            robot = await LiveRobot.create(dry_run=True)  # dry_run: connects and reads, never moves
            try:
                return await robot.snapshot()
            finally:
                await robot.close()

        frame = asyncio.run(snap())
        print("depth + camera pose saved for 3D later:", save_frame(frame))
        image = frame.color
    else:
        image = cv2.imread(str(args.image))
        if image is None:
            sys.exit(f"could not read {args.image}")

    start = time.perf_counter()
    detector = CanDetector()
    print(f"YOLO ready in {time.perf_counter() - start:.1f} s")

    out_dir = SCANS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    items = scan(image, detector, out_dir)
    counts = Counter(it.label for it in items)
    print("\non the table:", dict(counts) or "nothing")
    print("recorded in", out_dir)


if __name__ == "__main__":
    main()
