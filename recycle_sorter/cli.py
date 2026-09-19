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
    ap.add_argument("--look-only", action="store_true",
                    help="take the first look (survey, and scan if the mode has one), report what was found, pick nothing")
    ap.add_argument("--look", choices=["once", "when_needed", "every_pick"],
                    help="how often to take a new picture (default: pick.look in workspace.yaml)")
    ap.add_argument("--want", nargs="+", metavar="CLASS[=N]",
                    help="fetch an order instead of sorting everything: e.g. --want coke=2 water")
    ap.add_argument("--order", metavar="TEXT",
                    help='the same, in words: --order "2 cokes and a sparkling water" (llm/parse_order.py, needs Ollama)')
    ap.add_argument("--cam-pos", action="store_true",
                    help="--mode label: read labels from the survey picture too, skipping the closer "
                    "`scan` picture (mode label_cam_pos)")
    ap.add_argument("--replay", type=Path, metavar="DIR", help="run perception over saved frames; no robot")
    ap.add_argument(
        "--action",
        choices=["reset", "open-gripper", "static-cycle", "go-to-pick", "go-to-place"],
        help="one-off action, no camera: `reset` returns the arm to the survey pose, `open-gripper` "
        "releases whatever it holds; the rest are blind fixed-pose motion checks (same poses as move_arm.py)",
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

    wanted = None
    if args.want:
        wanted = {name: int(n or 1) for name, _, n in (w.partition("=") for w in args.want)}
    elif args.order:
        from dotenv import load_dotenv

        load_dotenv()  # OLLAMA_URL, before the parser reads it
        from llm.parse_order import parse_order

        order = parse_order(args.order)
        wanted = dict(order.items)
        print("order:", wanted, "| not available:", order.not_supported)
        if not wanted:
            return

    if args.cam_pos and args.mode == "label":
        args.mode = "label_cam_pos"
    robot = await LiveRobot.create(dry_run=args.dry_run, step=args.step)
    try:
        # In dry-run nothing moves, so one picture gives the whole plan.
        counts = await run_sort(robot, args.mode, args.max_picks, look="once" if args.dry_run else args.look, look_only=args.look_only, wanted=wanted)
        print("fetched:" if wanted else "sorted:", dict(counts))
    except (KeyboardInterrupt, asyncio.CancelledError):
        await robot.manip.stop()
        await robot.machine.stop_all()
        raise
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
