"""Full pipeline: typed command -> LLM -> order -> the arm fetches the items.

    $ python pipeline.py
    command> 2 cokes and a sparkling water
    order: {"coke": 2, "sparkling_water": 1}
    ... survey, YOLO boxes, local VLM labels, pick + place ...
    fetched: {"coke": 2, "sparkling_water": 1}

Per command:
  1. llm/parse_order.py   text -> {category: count} (qwen2.5:1.5b)
  2. recycle_sorter       run_sort(..., mode="label", wanted=order): survey picture -> YOLO boxes ->
                          llm/classify_image.py labels each crop (qwen2.5vl:7b + reference_pics/) ->
                          picks the surest match of each ordered category, one pile per category.
                          Positions come from the `survey` pose (where the grasp was calibrated);
                          labels from a closer picture at the `scan` pose (cam_lower_pos)
  3. result               fetched and missing counts printed. The latest look's two annotated
                          YOLO pictures are kept in data/scans/<launch time>/ as survey.png and
                          scan.png (overwritten each look). Nothing else is saved.

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
import sys
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()  # OLLAMA_URL / models, before the llm modules read the environment

from llm.parse_order import parse_order  # noqa: E402
from llm.warmup import warm_all  # noqa: E402
from recycle_sorter.app import run_sort  # noqa: E402
from recycle_sorter.config import DATA_DIR  # noqa: E402
from recycle_sorter.io.robot import LiveRobot  # noqa: E402

SCANS_DIR = DATA_DIR / "scans"


def _log_to_console() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    ours = logging.getLogger("recycle_sorter")  # not root: viam logs there too, and would print twice
    ours.addHandler(handler)
    ours.setLevel(logging.INFO)
    ours.propagate = False


async def handle(command: str, robot: LiveRobot, args) -> None:
    order = await asyncio.to_thread(parse_order, command)  # blocking HTTP call to Ollama
    print(f"order: {json.dumps(order.items)}"
          + (f"  (not available: {order.not_supported})" if order.not_supported else ""))
    if not order.items:
        print("nothing to fetch")
        return
    try:
        fetched = await run_sort(robot, args.mode, args.max_picks, look="once" if args.dry_run else args.look,
                                 wanted=dict(order.items), pictures=args.pictures)
    except (RuntimeError, ValueError) as e:  # e.g. no room for piles, camera download failed
        print(f"could not fetch the order: {e}")
        return
    missing = Counter(order.items) - Counter(fetched)
    print(("planned: " if args.dry_run else "fetched: ") + json.dumps(dict(fetched))
          + (f"  missing: {json.dumps(dict(missing))}" if missing else ""))


async def main() -> None:
    ap = argparse.ArgumentParser(description="Typed command -> order -> the arm fetches it.")
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
    # One folder per launch, holding only the latest look's two YOLO pictures (survey.png, scan.png).
    args.pictures = SCANS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    print("pictures:", args.pictures)

    # Load both models in the background while the robot connects and the order is typed: the
    # vision model, which also reads the reference photos (~55 s when cold), then the order model
    # (~2 s). Both stay loaded afterwards (keep_alive). Ollama queues requests, so an order or a
    # crop sent before the warm-up finishes simply waits for it.
    # One task, vision model first: loading the 7B model second makes Ollama unload the order model.
    warming = asyncio.create_task(asyncio.to_thread(warm_all))
    warming.add_done_callback(lambda t: t.exception() and print(f"model warm-up failed: {t.exception()}"))

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
