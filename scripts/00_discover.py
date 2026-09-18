"""Read-only. Prints what the machine actually has: resources, frames, camera streams, arm state.

    python scripts/00_discover.py
"""
import asyncio

from viam.components.arm import Arm
from viam.components.camera import Camera

from recycle_sorter.config import load_yaml
from recycle_sorter.io.robot import connect, find_camera_name


async def main() -> None:
    cfg = load_yaml("machine.yaml")
    machine = await connect()
    try:
        print("== resources ==")
        for r in sorted(machine.resource_names, key=lambda r: (r.type, r.subtype, r.name)):
            print(f"  {r.type:10s} {r.subtype:16s} {r.name}")

        print("\n== frame system (parent <- frame: translation) ==")
        for part in await machine.get_frame_system_config():
            f = part.frame
            p = f.pose_in_observer_frame
            print(f"  {p.reference_frame:12s} <- {f.reference_frame:16s} "
                  f"({p.pose.x:.1f}, {p.pose.y:.1f}, {p.pose.z:.1f}) "
                  f"o=({p.pose.o_x:.2f}, {p.pose.o_y:.2f}, {p.pose.o_z:.2f}) th={p.pose.theta:.1f}")

        name = find_camera_name(machine, cfg.get("camera"))
        cam = Camera.from_robot(machine, name)
        props = await cam.get_properties()
        i = props.intrinsic_parameters
        print(f"\n== camera '{name}' ==")
        print(f"  intrinsics {i.width_px}x{i.height_px} f=({i.focal_x_px:.1f}, {i.focal_y_px:.1f}) "
              f"c=({i.center_x_px:.1f}, {i.center_y_px:.1f})")
        images, _ = await cam.get_images()
        for img in images:
            print(f"  stream {img.name!r:10} mime={img.mime_type} bytes={len(img.data)}")
        print("  -> color and depth must be the same resolution (align_color_depth: true)")

        arm = Arm.from_robot(machine, cfg["arm"])
        pose = await arm.get_end_position()
        joints = await arm.get_joint_positions()
        print(f"\n== arm '{cfg['arm']}' ==")
        print(f"  end pose ({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f}) "
              f"o=({pose.o_x:.2f}, {pose.o_y:.2f}, {pose.o_z:.2f}) th={pose.theta:.1f}")
        print(f"  joints {[round(v, 1) for v in joints.values]}")
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
