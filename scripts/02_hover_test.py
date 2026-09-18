"""MOVES THE ARM. The calibration gate: does the robot reach where the camera says an object is?

Put ONE block in the pile area, then:

    python scripts/02_hover_test.py

It surveys, finds the block, and (after you press Enter for each move) hovers the open
gripper above it, fingers aligned to close across the block's short side. Measure with a ruler:

  * xy offset between gripper center and block center:  <= 15 mm passes.
    Bigger: the camera frame in the machine config is off, or `move_frame` in
    machine.yaml is wrong (try "arm" vs "gripper").
  * height: the fingertips should be --hover (60) mm above the block's top. If they
    are N mm too high, lower workspace.yaml -> gripper.tcp_offset by N (too low: raise it).
  * finger alignment: if the fingers are rotated relative to the block's short
    axis, put that angle in workspace.yaml -> gripper.yaw_offset_deg.
"""
import argparse
import asyncio
import logging

from recycle_sorter.app import save_debug
from recycle_sorter.io.recorder import save_frame
from recycle_sorter.io.robot import LiveRobot
from recycle_sorter.manipulation.pickplace import grasp_pose
from recycle_sorter.perception.segment import segment
from recycle_sorter.perception.select import choose_next
from recycle_sorter.types import Classification


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hover", type=float, default=60.0, help="mm above the object's top")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    robot = await LiveRobot.create(step=True)  # always confirm each move here
    m = robot.manip
    try:
        await m.goto_named("survey")
        frame = await robot.snapshot()
        save_frame(frame)
        objects = segment(frame, m.workspace)
        print("overlay:", save_debug(frame, objects, [Classification("", 0)] * len(objects), "hover-test", m.workspace))
        target = choose_next(objects, m.workspace)
        if target is None:
            raise SystemExit(f"found {len(objects)} object(s), none graspable - check table_top and unsorted_zone")

        x, y, _, theta = grasp_pose(target, m.workspace)
        print(f"object at ({x:.0f}, {y:.0f}) top z={target.top_z:.0f} "
              f"size {target.width:.0f}x{target.length:.0f} mm, yaw {target.yaw_deg:.0f}")
        await m.open()
        await m.move_to(x, y, target.top_z + args.hover, theta)
        await asyncio.to_thread(input, "Measure the offset now. ENTER to return to survey ")
        await m.goto_named("survey")
    except (KeyboardInterrupt, asyncio.CancelledError):
        await m.stop()
        raise
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
