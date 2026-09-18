"""Read-only on the robot. Jog the arm (Viam app CONTROL tab), then record where it is.

    python scripts/01_teach_pose.py home          # joint pose
    python scripts/01_teach_pose.py survey        # joint pose: camera top-down, ~400 mm above the pile
    python scripts/01_teach_pose.py --pile pile_1 # center of a sorted pile, from the gripper's current x/y
"""
import argparse
import asyncio

from viam.components.arm import Arm
from viam.services.motion import MotionClient

from recycle_sorter.config import load_yaml, save_yaml
from recycle_sorter.io.robot import connect


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="joint pose name, e.g. home or survey")
    ap.add_argument("--pile", help="record the current gripper x/y as the center of this sorted pile")
    args = ap.parse_args()
    if bool(args.name) == bool(args.pile):
        ap.error("give either a pose name or --pile NAME")

    cfg = load_yaml("machine.yaml")
    poses = load_yaml("poses.yaml")
    machine = await connect()
    try:
        if args.pile:
            motion = MotionClient.from_robot(machine, cfg["motion"])
            p = (await motion.get_pose(cfg["move_frame"], "world")).pose
            poses.setdefault("piles", {})[args.pile] = {"x": round(p.x, 1), "y": round(p.y, 1)}
            print(f"pile {args.pile}: x={p.x:.1f} y={p.y:.1f}  ({cfg['move_frame']} frame in world)")
        else:
            joints = await Arm.from_robot(machine, cfg["arm"]).get_joint_positions()
            poses.setdefault("joints", {})[args.name] = [round(v, 3) for v in joints.values]
            print(f"pose {args.name}: {poses['joints'][args.name]}")
        save_yaml("poses.yaml", poses)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
