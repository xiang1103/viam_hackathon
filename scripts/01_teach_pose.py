"""Read-only on the robot. Jog the arm (Viam app CONTROL tab), then record where it is.

    python scripts/01_teach_pose.py home          # where the arm parks when a run ends
    python scripts/01_teach_pose.py survey        # where the wrist camera sees the whole unsorted zone
    python scripts/01_teach_pose.py --area left   # run at two opposite corners of free table space:
                                                  # the rectangle between them becomes a sorted area
"""
import argparse
import asyncio

from viam.services.motion import MotionClient

from recycle_sorter.config import load_yaml, save_yaml
from recycle_sorter.io.robot import connect


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="joint pose name, e.g. home or survey")
    ap.add_argument("--area", help="record the gripper x/y as a corner of this sorted area (needs two corners)")
    args = ap.parse_args()
    if bool(args.name) == bool(args.area):
        ap.error("give either a pose name or --area NAME")

    cfg = load_yaml("machine.yaml")
    poses = load_yaml("poses.yaml")
    machine = await connect()
    try:
        if args.area:
            motion = MotionClient.from_robot(machine, cfg["motion"])
            p = (await motion.get_pose(cfg["move_frame"], "world")).pose
            corners = poses.setdefault("area_corners", {}).setdefault(args.area, [])
            if len(corners) >= 2:
                corners.clear()  # a third call starts the area over
            corners.append([round(p.x, 1), round(p.y, 1)])
            if len(corners) == 2:
                (x0, y0), (x1, y1) = corners
                poses.setdefault("sorted_areas", {})[args.area] = {"x": sorted([x0, x1]), "y": sorted([y0, y1])}
                print(f"area {args.area}: {poses['sorted_areas'][args.area]}")
            else:
                print(f"area {args.area}: corner 1 at ({p.x:.1f}, {p.y:.1f}) - now jog to the opposite corner and run again")
        else:
            motion = MotionClient.from_robot(machine, cfg["motion"])
            p = (await motion.get_pose(cfg["move_frame"], "world")).pose
            fields = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")
            poses.setdefault("named", {})[args.name] = {k: round(getattr(p, k), 4) for k in fields}
            print(f"pose {args.name}: {poses['named'][args.name]}")
        save_yaml("poses.yaml", poses)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
