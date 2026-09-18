"""MOVES THE ARM. Measures how far the arm can really reach over the table, gripper pointing down.

CLEAR THE TABLE FIRST - the gripper travels outward at can-grasp height along each direction.

    python scripts/05_probe_reach.py                 # Enter before every move
    python scripts/05_probe_reach.py --no-step        # once you trust it
    python scripts/05_probe_reach.py --z 250          # probe at carry height instead

Along each direction it steps outward from --start, asking the motion planner for every step.
A step the planner refuses moves nothing, so the last step that worked is the reach in that
direction. The safety bounds in config/workspace.yaml still apply: a direction can also end at
a bound, which is reported as such (the arm might reach further; the table or wall does not).

Put the result in config/workspace.yaml -> sorted_layout.max_reach.
"""
import argparse
import asyncio
import logging
import math

from recycle_sorter.io.robot import LiveRobot
from recycle_sorter.manipulation.safety import UnsafeTarget


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=float, nargs="+", default=[10, 0, -15, -30, -45, -60],
                    help="directions to probe, degrees from straight ahead (+x); negative = toward -y, the open side")
    ap.add_argument("--start", type=float, default=350, help="mm from the arm base to begin at")
    ap.add_argument("--step-mm", type=float, default=25)
    ap.add_argument("--z", type=float, default=None, help="fingertip height in mm (default: where a 122 mm can is grasped)")
    ap.add_argument("--no-step", action="store_true", help="do not wait for Enter before each move")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    robot = await LiveRobot.create(step=not args.no_step)
    m = robot.manip
    z = args.z if args.z is not None else m.workspace["table_top"] + 122 - m.workspace["gripper"]["finger_depth"]
    results = []
    try:
        await m.open()
        for angle in args.angles:
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            reached, why = None, "planner refused the first step"
            r = args.start
            while True:
                x, y = r * c, r * s
                try:
                    await m.move_to(x, y, z)
                except UnsafeTarget as e:
                    why = f"stopped at a safety bound ({e})"
                    break
                except Exception as e:
                    why = f"planner refused {r:.0f} mm ({type(e).__name__})"
                    break
                reached = r
                r += args.step_mm
            results.append((angle, reached, why))
            print(f"  direction {angle:+.0f} deg: reached {reached} mm - {why}")
            if reached:  # come back in before swinging to the next direction
                await m.move_to(args.start * c, args.start * s, z)
        await m.goto_named("survey")
    except (KeyboardInterrupt, asyncio.CancelledError):
        await m.stop()
        raise
    finally:
        await robot.close()

    print(f"\nreach with the fingertips at z = {z:.0f} mm:")
    for angle, reached, why in results:
        print(f"  {angle:+4.0f} deg   {str(reached) + ' mm' if reached else '-':>8}   {why}")
    by_planner = [r for _, r, why in results if r and "planner" in why]
    if by_planner:
        print(f"\nsuggested sorted_layout.max_reach: {min(by_planner) - 20:.0f}  (shortest planner-limited direction, minus 20 mm)")
    else:
        print("\nevery direction ended at a safety bound, not at the arm's limit: reach is not what limits you here.")


if __name__ == "__main__":
    asyncio.run(main())
