"""Read-only on the robot. Jog the arm (Viam app CONTROL tab), then record where it is.

    python scripts/01_teach_pose.py home          # joint pose
    python scripts/01_teach_pose.py survey        # joint pose: camera top-down, ~400 mm above the pile
    python scripts/01_teach_pose.py --bin bin_a   # x/y drop location, from the gripper's current position
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
    ap.add_argument("--bin", help="record the current gripper x/y as this bin's drop location")
    args = ap.parse_args()
    if bool(args.name) == bool(args.bin):
        ap.error("give either a pose name or --bin NAME")

    cfg = load_yaml("machine.yaml")
    poses = load_yaml("poses.yaml")
    machine = await connect()
    try:
        if args.bin:
            motion = MotionClient.from_robot(machine, cfg["motion"])
            p = (await motion.get_pose(cfg["move_frame"], "world")).pose
            poses.setdefault("bins", {})[args.bin] = {"x": round(p.x, 1), "y": round(p.y, 1)}
            print(f"bin {args.bin}: x={p.x:.1f} y={p.y:.1f}  ({cfg['move_frame']} frame in world)")
        else:
            joints = await Arm.from_robot(machine, cfg["arm"]).get_joint_positions()
            poses.setdefault("joints", {})[args.name] = [round(v, 3) for v in joints.values]
            print(f"pose {args.name}: {poses['joints'][args.name]}")
        save_yaml("poses.yaml", poses)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
