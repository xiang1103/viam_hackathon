"""Read-only. Prints what the machine actually has: resources, frames, camera streams, vision results, arm state.

    python scripts/00_discover.py
"""
import asyncio

from viam.components.arm import Arm
import numpy as np
from viam.components.camera import Camera
from viam.services.vision import VisionClient

from recycle_sorter.config import load_yaml
from recycle_sorter.io.robot import connect, find_camera_name
from recycle_sorter.perception.pcd import parse_pcd


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

        print("\n== vision services (machine.yaml -> vision.segmenters) ==")
        for label, service_name in cfg.get("vision", {}).get("segmenters", {}).items():
            try:
                pcos = await VisionClient.from_robot(machine, service_name).get_object_point_clouds(name)
            except Exception as e:
                print(f"  {label!r} -> {service_name}: FAILED ({e})")
                continue
            print(f"  {label!r} -> {service_name}: {len(pcos)} object(s) right now")
            for pco in pcos:
                raw = parse_pcd(pco.point_cloud)
                g = pco.geometries.geometries[0] if pco.geometries.geometries else None
                center = f"({g.center.x:.0f}, {g.center.y:.0f}, {g.center.z:.0f})" if g else "-"
                print(f"    frame={pco.geometries.reference_frame!r} label={g.label if g else ''!r} "
                      f"box center={center} points={len(raw)} point-cloud centroid(mm)={np.round(raw.mean(axis=0)) if len(raw) else '-'}")
        print("  -> box center and point-cloud centroid should roughly agree (same frame, both mm)")

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
