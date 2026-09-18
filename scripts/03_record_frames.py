"""Read-only. Save RGB-D snapshots from wherever the arm currently is, for offline perception work.

    python scripts/03_record_frames.py            # one frame -> data/frames/<timestamp>/
    python scripts/03_record_frames.py -n 5       # five frames, Enter between each (rearrange the pile)
    python scripts/03_record_frames.py --table    # BARE table: where the camera sees it, and whether depth is aligned

Then:  python -m recycle_sorter.cli --replay data/frames
"""
import argparse
import asyncio

from recycle_sorter.io.recorder import save_frame
from recycle_sorter.io.robot import LiveRobot
from recycle_sorter.perception.segment import table_report


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    robot = await LiveRobot.create(dry_run=True)  # dry_run: this script never moves anything
    try:
        for i in range(args.n):
            if i:
                await asyncio.to_thread(input, "rearrange, then ENTER ")
            frame = await robot.snapshot()
            print("saved", save_frame(frame))
            if args.table:
                print(table_report(frame, robot.manip.workspace))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
