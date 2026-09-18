"""Find cans and bottles in a picture with YOLO. Load once, then call it as often as you like.

In a pipeline:

    from detect_cans import CanDetector

    detector = CanDetector()            # loads the weights (~2 s) and warms the model up - do this ONCE
    boxes = detector(image_bgr)         # ~0.2 s per 1280x720 picture on this laptop's CPU
    for b in boxes:
        crop = image_bgr[b.y0:b.y1, b.x0:b.x1]      # the can, e.g. to read its label
        print(b.label, b.confidence, (b.x0, b.y0, b.x1, b.y1), b.mask is not None)

With the robot's depth picture, the same boxes become 3D positions and grasps:

    from recycle_sorter.perception.yolo import observations_from_boxes
    objects = observations_from_boxes(boxes, frame, workspace)     # .centroid (mm), .height, .grasp_xy, .crop

From the command line:

    python detect_cans.py data/frames/<timestamp>/color.png       # writes <name>.detected.jpg next to it
    python detect_cans.py --camera                                 # live picture from the robot (moves nothing)

YOLO is only the boundary finder here. It calls everything "bottle" or "can" - ignore that; what
an item IS comes from reading its label (recycle_sorter/classify/brand_claude.py).

The model is YOLOE, an open-vocabulary YOLO: it looks for the words in `classes` rather than a
fixed list, so no training is needed. Standard pretrained YOLO has no "can" class. Settings live
in config/machine.yaml under `yolo`. First run downloads the weights into models/ (28 MB, plus a
600 MB text encoder that is only needed that one time).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from recycle_sorter.config import load_yaml
from recycle_sorter.perception.yolo import Box, YoloDetector


class CanDetector:
    def __init__(self, settings: dict | None = None, warm_up: bool = True):
        self.settings = settings or load_yaml("machine.yaml")["yolo"]
        self._yolo = YoloDetector(self.settings)
        self._yolo.model  # load the weights now, not in the middle of the first real picture
        if warm_up:
            # The first pass through a freshly loaded network is several times slower than the rest.
            size = self.settings.get("imgsz", 1280)
            self._yolo.detect(np.zeros((size * 9 // 16, size, 3), np.uint8))

    def __call__(self, image_bgr: np.ndarray) -> list[Box]:
        """Boxes (and outlines, as `.mask`) for every can or bottle, best first."""
        return sorted(self._yolo.detect(image_bgr), key=lambda b: -b.confidence)


def draw(image_bgr: np.ndarray, boxes: list[Box]) -> np.ndarray:
    out = image_bgr.copy()
    for i, b in enumerate(boxes):
        if b.mask is not None:
            out[b.mask] = (0.55 * out[b.mask] + 0.45 * np.array([255, 200, 0])).astype(np.uint8)
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (255, 200, 0), 2)
        cv2.putText(out, f"{i} {b.confidence:.2f}", (b.x0, max(b.y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Find cans and bottles in a picture.")
    ap.add_argument("image", nargs="?", type=Path, help="a color picture (png / jpg)")
    ap.add_argument("--camera", action="store_true", help="take a live picture from the robot instead (read-only)")
    args = ap.parse_args()
    if bool(args.image) == args.camera:
        ap.error("give a picture, or --camera")

    start = time.perf_counter()
    detector = CanDetector()
    print(f"model ready in {time.perf_counter() - start:.1f} s")

    frame = None
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
        out = save_frame(frame) / "color.detected.jpg"
        image = frame.color
    else:
        image = cv2.imread(str(args.image))
        if image is None:
            sys.exit(f"could not read {args.image}")
        out = args.image.with_suffix(".detected.jpg")

    start = time.perf_counter()
    boxes = detector(image)
    print(f"{len(boxes)} found in {time.perf_counter() - start:.2f} s")
    for i, b in enumerate(boxes):
        print(f"  {i}: {b.confidence:.2f}  box ({b.x0}, {b.y0})-({b.x1}, {b.y1})")

    if frame is not None:
        from recycle_sorter.perception.yolo import observations_from_boxes

        workspace = load_yaml("workspace.yaml")
        workspace["unsorted_zone"] = workspace["search_region"]
        print("in the robot's coordinates (mm from the arm base):")
        for o in observations_from_boxes(boxes, frame, workspace):
            reach = float(np.hypot(*o.centroid))
            print(f"  at ({o.centroid[0]:.0f}, {o.centroid[1]:.0f})  {o.height:.0f} mm tall  {reach:.0f} mm from the arm")

    cv2.imwrite(str(out), draw(image, boxes))
    print("picture:", out)


if __name__ == "__main__":
    main()
