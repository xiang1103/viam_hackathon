"""Full pipeline: typed command -> order -> camera picture -> YOLO boxes -> VLM labels -> picks.

    $ python pipeline.py --camera
    command> 2 cokes and a sparkling water
    {"command": "2 cokes and a sparkling water",
     "order": {"coke": 2, "sparkling_water": 1},
     "picks": [{"index": 5, "label": "coke", ...}, {"index": 7, ...}, {"index": 0, ...}],
     "missing": {}, ...}

Steps for every command:
  1. llm/parse_order.py      command text -> {category: count}          (qwen2.5:1.5b)
  2. camera / saved picture  a fresh picture, since the table changes between orders
  3. detect_cans.py          YOLO finds every can/bottle -> numbered boxes
  4. llm/classify_image.py   each box's crop -> category, with reference_pics/ (qwen2.5vl:7b)
  5. choose                  for each ordered category, the most confident matching boxes

Usage:
    python pipeline.py --camera                         # type commands; new picture each time
    python pipeline.py --image picture.jpg              # type commands against a saved picture
    python pipeline.py --image picture.jpg "2 cokes"    # one command, then exit

Every command is recorded in data/scans/<timestamp>/ (picture, crops, detected.jpg,
results.json from scan_drinks.py) plus order.json with this script's output.
Nothing here moves the arm: --camera connects read-only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detect_cans import CanDetector
from llm.classify_image import load_references
from llm.parse_order import parse_order
from scan_drinks import SCANS_DIR, ScannedItem, scan

# Only pick items the VLM is at least this sure about; the rest count as missing.
MIN_CONFIDENCE = "medium"
_RANK = {"high": 0, "medium": 1, "low": 2}


def take_picture() -> np.ndarray:
    """A live color picture from the robot's camera. Read-only: the arm never moves."""
    from recycle_sorter.io.recorder import save_frame
    from recycle_sorter.io.robot import LiveRobot

    async def snap():
        robot = await LiveRobot.create(dry_run=True)
        try:
            return await robot.snapshot()
        finally:
            await robot.close()

    frame = asyncio.run(snap())
    save_frame(frame)  # depth + camera pose, for the 3D grasp step later
    return frame.color


def choose(order: dict[str, int], items: list[ScannedItem],
           min_confidence: str = MIN_CONFIDENCE) -> tuple[list[ScannedItem], dict[str, int]]:
    """For each ordered category, the best matching items (VLM confidence, then YOLO's).
    Returns (picks, missing) where missing = {category: how many could not be found}."""
    picks, missing = [], {}
    for label, count in order.items():
        matches = sorted(
            (it for it in items
             if it.label == label and _RANK[it.confidence] <= _RANK[min_confidence]),
            key=lambda it: (_RANK[it.confidence], -it.yolo_confidence))
        picks += matches[:count]
        if len(matches) < count:
            missing[label] = count - len(matches)
    return picks, missing


def run_command(command: str, image: np.ndarray, detector: CanDetector) -> dict:
    start = time.perf_counter()
    order = parse_order(command)
    print(f"order: {order.items}" + (f"  not available: {order.not_supported}"
                                     if order.not_supported else "")
          + f"  [{time.perf_counter() - start:.1f} s]", file=sys.stderr)

    result = {"command": command, "order": order.items, "not_supported": order.not_supported,
              "picks": [], "missing": {}, "detections": []}
    if not order.items:  # nothing we stock was asked for: no need to look at the table
        return result

    out_dir = SCANS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    items = scan(image, detector, out_dir, log=lambda msg: print(msg, file=sys.stderr))
    picks, missing = choose(order.items, items)

    def row(it: ScannedItem) -> dict:
        return {"index": it.index, "label": it.label, "confidence": it.confidence,
                "bbox": list(it.bbox), "yolo_confidence": it.yolo_confidence}

    result.update(picks=[row(it) for it in picks], missing=missing,
                  detections=[row(it) for it in items], scan_dir=str(out_dir))
    (out_dir / "order.json").write_text(json.dumps(result, indent=2))
    print(f"total {time.perf_counter() - start:.1f} s, recorded in {out_dir}", file=sys.stderr)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Command -> order -> YOLO + VLM -> which boxes to pick.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true", help="live picture from the robot (read-only)")
    src.add_argument("--image", type=Path, help="a saved color picture (png / jpg)")
    ap.add_argument("command", nargs="*", help="one command to run, then exit; omit to type commands")
    args = ap.parse_args()

    image = None
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            sys.exit(f"could not read {args.image}")

    print("loading YOLO and reference photos...", file=sys.stderr)
    detector = CanDetector()
    print(f"{len(load_references())} reference photos", file=sys.stderr)

    def handle(command: str) -> None:
        picture = take_picture() if args.camera else image
        print(json.dumps(run_command(command, picture, detector), indent=2))

    if args.command:
        handle(" ".join(args.command))
        return
    print("Type a command (e.g. '2 cokes and a sparkling water'); empty line or 'quit' to exit.",
          file=sys.stderr)
    while True:
        try:
            command = input("command> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if not command or command.lower() in ("quit", "exit", "q"):
            break
        handle(command)


if __name__ == "__main__":
    main()
