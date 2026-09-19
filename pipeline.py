"""Full pipeline: typed command -> LLM -> order JSON -> the arm fetches the items.

    $ python pipeline.py
    command> 2 cokes and a sparkling water
    order: {"coke": 2, "sparkling_water": 1}   (saved to data/orders/<timestamp>.json)
    ... survey, YOLO boxes, local VLM labels, pick + place ...
    fetched: {"coke": 2, "sparkling_water": 1}

Per command:
  1. llm/parse_order.py   text -> {category: count} (qwen2.5:1.5b), written to data/orders/<ts>.json
  2. recycle_sorter       run_sort(..., mode="label", wanted=order): survey picture -> YOLO boxes ->
                          llm/classify_image.py labels each crop (qwen2.5vl:7b + reference_pics/) ->
                          picks the surest match of each ordered category, one pile per category.
                          Positions come from the `survey` pose (where the grasp was calibrated);
                          labels from a closer picture at the `scan` pose (cam_lower_pos)
  3. result               fetched and missing counts added to the same JSON file, and the
                          last annotated camera picture (boxes + labels) copied to
                          data/scans/<timestamp>.png

Usage:
    python pipeline.py                     # type commands; THE ARM MOVES
    python pipeline.py --step              # press Enter before every arm motion (first runs)
    python pipeline.py --dry-run           # plan and log every move, move nothing
    python pipeline.py "2 cokes"           # one command, then exit
    python pipeline.py --cam-pos           # labels from the survey picture too: no closer `scan` picture

Needs `ollama serve` with qwen2.5:1.5b and qwen2.5vl:7b, and the robot (.env).
Image-only check with no arm: python scan_drinks.py --camera
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()  # OLLAMA_URL / models, before the llm modules read the environment

from llm.classify_image import warm_up  # noqa: E402
from llm.parse_order import parse_order  # noqa: E402
from recycle_sorter.app import run_sort  # noqa: E402
from recycle_sorter.config import DATA_DIR  # noqa: E402
from recycle_sorter.io.robot import LiveRobot  # noqa: E402

ORDERS_DIR = DATA_DIR / "orders"
SCANS_DIR = DATA_DIR / "scans"
# run_sort draws every picture it takes, boxes and labels on, as data/debug/<frame timestamp>.png,
# and the closer label picture (view: scan) with each item's crop box as ...-scan.png
_LOOK_PICTURE = re.compile(r"\d{8}-\d{6}-\d+(-scan)?\.png")


def _log_to_console() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    ours = logging.getLogger("recycle_sorter")  # not root: viam logs there too, and would print twice
    ours.addHandler(handler)
    ours.setLevel(logging.INFO)
    ours.propagate = False


def _keep_last_picture(record: dict, since: float) -> None:
    """Copy the last annotated survey picture of this order to data/scans/<time>.png, and the
    last label (scan) picture, with each can's crop box, to data/scans/<time>-scan.png.
    The links of that scan picture (which boxes each read came from) go into the record."""
    looks = [p for p in (DATA_DIR / "debug").glob("*.png")
             if _LOOK_PICTURE.fullmatch(p.name) and p.stat().st_mtime >= since]
    for suffix, key in (("", "picture"), ("-scan", "scan_picture")):
        kind = [p for p in looks if p.stem.endswith("-scan") == bool(suffix)]
        if kind:
            SCANS_DIR.mkdir(parents=True, exist_ok=True)
            out = SCANS_DIR / f"{record['time']}{suffix}.png"
            shutil.copyfile(max(kind, key=lambda p: p.stat().st_mtime), out)
            record[key] = str(out)
            print(f"{key.replace('_', ' ')}:", out)
    # Which survey box and scan box each can's read came from, for the last scan picture.
    links = [p for p in (DATA_DIR / "debug").glob("*-scan-links.json") if p.stat().st_mtime >= since]
    if links:
        record["links"] = json.loads(max(links, key=lambda p: p.stat().st_mtime).read_text())


def _save(record: dict) -> None:
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    (ORDERS_DIR / f"{record['time']}.json").write_text(json.dumps(record, indent=2))


async def handle(command: str, robot: LiveRobot, args) -> dict:
    order = await asyncio.to_thread(parse_order, command)  # blocking HTTP call to Ollama
    record = {"time": datetime.now().strftime("%Y%m%d-%H%M%S"), "command": command,
              "items": order.items, "not_supported": order.not_supported}
    _save(record)
    print(f"order: {json.dumps(order.items)}"
          + (f"  (not available: {order.not_supported})" if order.not_supported else "")
          + f"  -> {ORDERS_DIR / (record['time'] + '.json')}")
    if not order.items:
        print("nothing to fetch")
        return record

    started = time.time()
    try:
        fetched = await run_sort(robot, args.mode, args.max_picks,
                                 look="once" if args.dry_run else args.look, wanted=dict(order.items))
    except (RuntimeError, ValueError) as e:  # e.g. no room for piles, camera download failed
        record.update(result="failed", error=str(e))
        _keep_last_picture(record, started)
        _save(record)
        print(f"could not fetch the order: {e}")
        return record
    missing = Counter(order.items) - Counter(fetched)
    record.update(result="planned" if args.dry_run else "done",
                  fetched=dict(fetched), missing=dict(missing))
    _keep_last_picture(record, started)
    _save(record)
    print(("planned: " if args.dry_run else "fetched: ") + json.dumps(dict(fetched))
          + (f"  missing: {json.dumps(dict(missing))}" if missing else ""))
    return record


async def main() -> None:
    ap = argparse.ArgumentParser(description="Typed command -> order JSON -> the arm fetches it.")
    ap.add_argument("command", nargs="*", help="one command to run, then exit; omit to type commands")
    ap.add_argument("--dry-run", action="store_true", help="perceive and log planned moves; move nothing")
    ap.add_argument("--step", action="store_true", help="press Enter before every arm motion")
    ap.add_argument("--mode", default="label", help="classifier in config/sort.yaml (default: local VLM)")
    ap.add_argument("--look", choices=["once", "when_needed", "every_pick"],
                    help="how often to take a new picture (default: pick.look in workspace.yaml)")
    ap.add_argument("--max-picks", type=int, default=20)
    ap.add_argument("--cam-pos", action="store_true",
                    help="read labels from the survey picture too, skipping the closer `scan` picture "
                    "(mode label_cam_pos)")
    args = ap.parse_args()
    _log_to_console()

    if args.cam_pos and args.mode == "label":
        args.mode = "label_cam_pos"

    # Load the vision model and have it read the reference photos (~50 s when cold) while the
    # robot connects, the order is typed and the arm goes to survey - not when the first can is read.
    # Ollama queues requests, so a crop sent before this finishes simply waits for it.
    warming = asyncio.create_task(asyncio.to_thread(warm_up))
    warming.add_done_callback(lambda t: print(
        f"vision model warm-up failed: {t.exception()}" if t.exception() else f"vision model ready ({t.result():.0f} s)"))

    robot = await LiveRobot.create(dry_run=args.dry_run, step=args.step)
    try:
        if args.command:
            await handle(" ".join(args.command), robot, args)
            return
        print("Type a command (e.g. '2 cokes and a sparkling water'); empty line or 'quit' to exit.")
        while True:
            try:
                command = (await asyncio.to_thread(input, "command> ")).strip()
            except EOFError:
                break
            if not command or command.lower() in ("quit", "exit", "q"):
                break
            await handle(command, robot, args)
    except (KeyboardInterrupt, asyncio.CancelledError):
        await robot.manip.stop()  # Ctrl-C mid-pick: stop the arm before anything else
        await robot.machine.stop_all()
        raise
    finally:
        await robot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
