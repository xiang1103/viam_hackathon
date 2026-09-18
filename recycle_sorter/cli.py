from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .app import run_replay, run_sort
from .io.robot import LiveRobot
from .service import run_local


async def main() -> None:
    ap = argparse.ArgumentParser(prog="recycle_sorter", description="Sort a pile with a Viam arm.")
    ap.add_argument("--mode", default="color", help="which classifier / bin table to use (sort.yaml modes)")
    ap.add_argument("--dry-run", action="store_true", help="perceive and log planned moves; move nothing")
    ap.add_argument("--step", action="store_true", help="press Enter before every motion")
    ap.add_argument("--max-picks", type=int, default=50)
    ap.add_argument("--replay", type=Path, metavar="DIR", help="run perception over saved frames; no robot")
    ap.add_argument(
        "--action",
        choices=["static-cycle", "go-to-pick", "go-to-place"],
        help="blind fixed-pose motion check through this package, no camera (same poses as move_arm.py)",
    )
    args = ap.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    ours = logging.getLogger("recycle_sorter")  # not the root logger: viam logs there too, and would print twice
    ours.addHandler(handler)
    ours.setLevel(logging.INFO)
    ours.propagate = False

    if args.replay:
        await run_replay(args.replay, args.mode)
        return
    if args.action:
        print("Result:", await run_local({"action": args.action}, dry_run=args.dry_run, step=args.step))
        return

    robot = await LiveRobot.create(dry_run=args.dry_run, step=args.step)
    try:
        # In dry-run the arm never moves, so one pass shows the whole plan.
        counts = await run_sort(robot, args.mode, 1 if args.dry_run else args.max_picks)
        print("sorted:", dict(counts))
    except (KeyboardInterrupt, asyncio.CancelledError):
        await robot.manip.stop()
        await robot.machine.stop_all()
        raise
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
