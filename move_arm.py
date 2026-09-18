"""Blind pick -> place between two fixed poses. No camera: this checks arm, gripper, and motion.

    python move_arm.py                      # static-cycle
    python move_arm.py go-to-pick           # or go-to-place
    python move_arm.py --step               # press Enter before every motion
    python move_arm.py --dry-run            # log the moves, execute nothing

The logic lives in the package:
    recycle_sorter/service.py                  MyGenericService (the code-1 module) + do_command
    recycle_sorter/manipulation/pickplace.py   pick_at / place_at / static_cycle
    config/poses.yaml                          the pick and place poses
"""
import argparse
import asyncio
import logging

from recycle_sorter.service import MyGenericService, run_local  # noqa: F401  (MyGenericService: module entry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="static-cycle", choices=["static-cycle", "go-to-pick", "go-to-place"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--step", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    print("Result:", asyncio.run(run_local({"action": args.action}, dry_run=args.dry_run, step=args.step)))


if __name__ == "__main__":
    main()
