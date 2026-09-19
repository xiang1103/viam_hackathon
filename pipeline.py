"""Full pipeline: typed command -> LLM -> order -> the arm fetches the items.

    $ python pipeline.py
    command> 2 cokes and a sparkling water
    order: {"coke": 2, "water": 1}
    ... survey, YOLO boxes, local VLM labels, pick + place ...
    fetched: {"coke": 2, "water": 1}

Per command:
  1. llm/parse_order.py   text -> {category: count} as JSON (qwen2.5:1.5b). What is not a category we
                          stock is listed as not available and skipped.
  2. recycle_sorter       run_sort(..., mode="label", wanted=order):
                          a. `scan` pose (cam_lower_pos): a close picture from the side, where the labels
                             can be read. YOLO boxes each can; llm/classify_image.py labels each crop
                             (qwen2.5vl:7b + reference_pics/).
                          b. `survey` pose (top view): where each can IS. YOLO's outline of the can gives
                             the grasp point; each can is linked to its box in the scan picture. The
                             survey is taken last, so the arm goes from it straight to the pick.
                          c. picks the surest match of each ordered category, one pile per category.
                             A category nobody read off a can is skipped and reported as missing.
                          All of a command's items are fetched from that ONE look (--look once, the
                          default here): no new pictures between picks. A can that a pick nudged, or a
                          grasp that failed, is not re-checked - use --look when_needed for that.
  3. result               printed, and written to data/scans/<launch time>/ together with the latest
                          look's two annotated YOLO pictures: scan.png (labels), survey.png
                          (positions), order.json (the order, what was fetched, what was missing).

Usage:
    python pipeline.py                     # type commands; THE ARM MOVES
    python pipeline.py --step              # press Enter before every arm motion (first runs)
    python pipeline.py --dry-run           # plan and log every move, move nothing
    python pipeline.py "2 cokes"           # one command, then exit

Needs `ollama serve` with qwen2.5:1.5b and qwen2.5vl:7b, and the robot (.env).
Image-only check with no arm: python scan_drinks.py --camera
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # OLLAMA_URL / models, before the llm modules read the environment

from llm.parse_order import parse_order  # noqa: E402
from llm.warmup import warm_all  # noqa: E402
from recycle_sorter.app import run_sort  # noqa: E402
from recycle_sorter.config import DATA_DIR, load_yaml  # noqa: E402
from recycle_sorter.io.robot import LiveRobot  # noqa: E402
from recycle_sorter.perception.yolo import shared_detector  # noqa: E402

SCANS_DIR = DATA_DIR / "scans"


def _log_to_console() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    ours = logging.getLogger("recycle_sorter")  # not root: viam logs there too, and would print twice
    ours.addHandler(handler)
    ours.setLevel(logging.INFO)
    ours.propagate = False


async def handle(command: str, get_robot, args) -> None:
    """Parse first; wait for the robot connection only when there is something to fetch."""
    order = await asyncio.to_thread(parse_order, command)  # blocking HTTP call to Ollama
    print(f"order: {json.dumps(order.items)}"
          + (f"  (not available: {order.not_supported})" if order.not_supported else ""))
    if not order.items:
        print("nothing to fetch")
        return
    robot = await get_robot()
    try:
        # One look per command: the label picture and the position picture are taken once, and every item
        # of the order is then fetched from them - no new picture between picks, none at the end.
        fetched = await run_sort(robot, args.mode, args.max_picks, look="once" if args.dry_run else (args.look or "once"),
                                 wanted=dict(order.items), pictures=args.pictures)
    except (RuntimeError, ValueError) as e:  # e.g. no room for piles, camera download failed
        print(f"could not fetch the order: {e}")
        return
    missing = Counter(order.items) - Counter(fetched)
    print(("planned: " if args.dry_run else "fetched: ") + json.dumps(dict(fetched))
          + (f"  missing: {json.dumps(dict(missing))}" if missing else ""))
    save_result(args.pictures, command, order, fetched, missing, args.dry_run)


def save_result(folder: Path, command: str, order, fetched, missing, dry_run: bool) -> Path:
    """The command's outcome as JSON, next to the pictures it was decided from (one entry per command)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "order.json"
    history = json.loads(path.read_text()) if path.exists() else []
    history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "command": command,
        "order": dict(order.items),
        "not_available": list(order.not_supported),
        "planned" if dry_run else "fetched": dict(fetched),
        "missing": dict(missing),
    })
    path.write_text(json.dumps(history, indent=2))
    return path


async def main() -> None:
    ap = argparse.ArgumentParser(description="Typed command -> order -> the arm fetches it.")
    ap.add_argument("command", nargs="*", help="one command to run, then exit; omit to type commands")
    ap.add_argument("--dry-run", action="store_true", help="perceive and log planned moves; move nothing")
    ap.add_argument("--step", action="store_true", help="press Enter before every arm motion")
    ap.add_argument("--mode", default="label", help="classifier in config/sort.yaml (default: local VLM)")
    ap.add_argument("--look", choices=["once", "when_needed", "every_pick"],
                    help="how often to take a new picture. Default `once`: one look per command, and everything "
                    "ordered is fetched from it. `when_needed` looks again after a failed grasp or after picking a "
                    "can that had a close neighbour; `every_pick` after every pick")
    ap.add_argument("--max-picks", type=int, default=20)
    ap.add_argument("--cam-pos", action="store_true",
                    help="read labels from the survey picture too, skipping the closer `scan` picture "
                    "(mode label_cam_pos)")
    args = ap.parse_args()
    _log_to_console()

    if args.cam_pos and args.mode == "label":
        args.mode = "label_cam_pos"
        print("WARNING --cam-pos: the survey is a top view now (poses.yaml) - it shows lids, not labels, so labels "
              "read from it are guesses. Leave --cam-pos off to read them from the `scan` pose.")
    # One folder per launch, holding only the latest look's two YOLO pictures (survey.png, scan.png).
    args.pictures = SCANS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    print("pictures:", args.pictures)

    # Load both models in the background while the robot connects and the order is typed: the
    # vision model, which also reads the reference photos (~55 s when cold), then the order model
    # (~2 s). Both stay loaded afterwards (keep_alive). Ollama queues requests, so an order or a
    # crop sent before the warm-up finishes simply waits for it.
    # One task, vision model first: loading the 7B model second makes Ollama unload the order model.
    # Silent: anything printed here would land in the middle of the `command>` prompt. If Ollama is
    # down, the first command reports it anyway.
    # A daemon thread, not asyncio.to_thread: quitting must not wait for a ~55 s model load. Ollama
    # finishes loading it anyway, so it is still warm next time.
    def warm_quietly() -> None:
        try:
            warm_all(lambda msg: None)
        except Exception:
            pass

    threading.Thread(target=warm_quietly, name="model-warm-up", daemon=True).start()

    # YOLO too (~2 s: load, then a first prediction that is ~1 s slower than later ones), in its own
    # thread so it overlaps the Ollama load. Both pictures of every order then use this one copy
    # (shared_detector); an order that arrives before it is ready waits for it rather than loading twice.
    machine_cfg = load_yaml("machine.yaml")
    if machine_cfg.get("perception") == "yolo":
        # Silent like the Ollama warm-up: ultralytics' own messages (settings, downloads) would land in
        # the `command>` prompt. Read by ultralytics when it is imported, i.e. in the thread below.
        os.environ.setdefault("YOLO_VERBOSE", "False")
        def warm_yolo() -> None:
            try:
                shared_detector(machine_cfg["yolo"]).warm_up()
            except Exception:
                pass  # e.g. ultralytics missing: the first look reports it

        threading.Thread(target=warm_yolo, name="yolo-warm-up", daemon=True).start()

    # Connect to the robot in the background too (3-9 s through the Viam cloud), so the prompt shows
    # at once; the first command waits for the connection only if it is not done yet. Viam's INFO
    # lines ("Connecting to socket ...") would land in the prompt, so only its warnings get through.
    os.environ.setdefault("VIAM_LOG_LEVEL", "WARNING")  # read by connect.py; the SDK resets its level on connect
    connecting = asyncio.create_task(LiveRobot.create(dry_run=args.dry_run, step=args.step))
    robot = None

    async def get_robot() -> LiveRobot:
        nonlocal robot
        if robot is None:
            try:
                robot = await connecting
            except Exception as e:
                raise SystemExit(f"could not connect to the robot: {e}") from e
        return robot

    async def run(command: str) -> None:
        await handle(command, get_robot, args)

    try:
        if args.command:
            await run(" ".join(args.command))
            return
        print("Type a command (e.g. '2 cokes and a sparkling water'); empty line or 'quit' to exit.")
        while True:
            try:
                command = (await asyncio.to_thread(input, "command> ")).strip()
            except EOFError:
                break
            if not command or command.lower() in ("quit", "exit", "q"):
                break
            await run(command)
    except (KeyboardInterrupt, asyncio.CancelledError):
        if robot is not None:
            await robot.manip.stop()  # Ctrl-C mid-pick: stop the arm before anything else
            await robot.machine.stop_all()
        raise
    finally:
        if robot is None and not connecting.done():
            connecting.cancel()  # quit before it connected
        elif robot is None and not connecting.cancelled() and connecting.exception() is None:
            robot = connecting.result()  # connected, but no command was run
        if robot is not None:
            await robot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
