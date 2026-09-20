"""Demo prep: read every can's label ONCE, now, and save it for demo_pipeline.py.

    python demo_capture.py            # THE ARM MOVES: survey -> scan -> survey
    python demo_capture.py --step     # press Enter before each of those two moves

Takes the same look as one pipeline.py command (scan picture for the labels, survey picture for the
positions, YOLO on both, the local VLM on every can) and writes each can's position and label to
data/demo_labels.json, with the two annotated pictures next to it in data/demo_capture/. It ends at
the survey pose, ready for the demo. Nothing is picked.

Leave the cans where they are afterwards: demo_pipeline.py matches each can it sees to this file by
position, so a can that was moved is read by the VLM again.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from recycle_sorter.app import make_classifier, save_scan_debug, save_survey_picture  # noqa: E402
from recycle_sorter.config import DATA_DIR, load_yaml  # noqa: E402
from recycle_sorter.io.robot import LiveRobot  # noqa: E402
from recycle_sorter.policy.zones import resolve  # noqa: E402

LABELS_FILE = DATA_DIR / "demo_labels.json"
PICTURES = DATA_DIR / "demo_capture"


async def main() -> None:
    ap = argparse.ArgumentParser(description="Read every can's label once and save it for demo_pipeline.py.")
    ap.add_argument("--step", action="store_true", help="press Enter before every arm motion")
    ap.add_argument("--mode", default="label")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("recycle_sorter")
    log.setLevel(logging.INFO)

    classifier = make_classifier(args.mode, load_yaml("sort.yaml"))
    if not getattr(classifier, "needs_scan", False):
        raise SystemExit(f"mode {args.mode!r} does not read labels from the scan picture")
    robot = await LiveRobot.create(step=args.step)
    try:
        await robot.manip.goto_named("scan")
        scan = await robot.snapshot()
        await robot.manip.goto_named("survey")
        survey = await robot.snapshot()

        async def find(ws: dict[str, Any]):
            robot.manip.workspace = ws
            return await robot.detect(survey)

        workspace, _ = await resolve(robot.manip.workspace, find)
        robot.manip.workspace = workspace
        objects = await robot.detect(survey)
        if not objects:
            raise SystemExit("no cans found in the survey picture")
        print(f"{len(objects)} can(s) found; reading labels (~10 s each)...")
        results = await classifier.classify_scan(objects, scan, survey, workspace)

        save_scan_debug(scan, classifier.last_views, results, "scan", classifier.last_scan_only, out_dir=PICTURES)
        save_survey_picture(survey, objects, results, robot.last_rejected, "survey", workspace, PICTURES)
        cans = [{"x": float(o.centroid[0]), "y": float(o.centroid[1]), "label": r.label,
                 "confidence": r.confidence, "meta": r.meta} for o, r in zip(objects, results)]
        LABELS_FILE.write_text(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "cans": cans},
                                          indent=2, default=str))
        for i, c in enumerate(cans):
            print(f"  #{i} {c['label']:<18} {c['confidence']:.2f} at ({c['x']:.0f}, {c['y']:.0f})"
                  f"  {c['meta'].get('visible_text', '')}")
        print(f"saved {LABELS_FILE}; pictures in {PICTURES}")
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
