from __future__ import annotations

import logging

from ..types import ObjectObservation
from .motion import Manipulator

log = logging.getLogger(__name__)


async def pick_at(m: Manipulator, x: float, y: float, z: float, theta: float = 0.0) -> bool:
    """Open, approach from above, descend, grasp, raise clear. z is fingertip height.

    Returns grab()'s verdict on whether something is held.
    """
    p = m.workspace["pick"]
    await m.open()
    try:
        await m.move_to(x, y, z + p["approach"], theta)
    except Exception as e:
        # A 180 degree flip is the same grasp for a parallel gripper, but can be
        # reachable when the first wrist angle is not.
        log.warning("approach failed (%s); retrying with theta+180", e)
        theta += 180.0
        await m.move_to(x, y, z + p["approach"], theta)
    await m.move_to(x, y, z, theta, linear=True)
    held = await m.grab()
    # Raise before any lateral move so the object does not drag across the table.
    await m.move_to(x, y, z + p["lift"], theta, linear=True)
    return held


async def place_at(m: Manipulator, x: float, y: float, z: float, theta: float = 0.0) -> None:
    """Arrive at carry height, descend, release, lift clear. z is fingertip height."""
    p = m.workspace["pick"]
    await m.move_to(x, y, z + p["lift"], theta)
    await m.move_to(x, y, z, theta, linear=True)
    await m.open()
    await m.move_to(x, y, z + p["lift"], theta, linear=True)


def grasp_pose(obs: ObjectObservation, workspace: dict) -> tuple[float, float, float, float]:
    """Top-down grasp: (x, y, z, theta). Fingers close across the object's short axis."""
    g = workspace["gripper"]
    z = max(obs.top_z - g["finger_depth"], workspace["table_top"] + workspace["bounds"]["z_min_above_table"])
    return float(obs.centroid[0]), float(obs.centroid[1]), z, obs.yaw_deg + g["yaw_offset_deg"]


async def pick(m: Manipulator, obs: ObjectObservation) -> bool:
    return await pick_at(m, *grasp_pose(obs, m.workspace))


async def place(m: Manipulator, x: float, y: float, grasp_z: float) -> None:
    """Set the item down at a pile slot (see policy/piles.py).

    It was grasped with the fingertips at grasp_z while resting on the table, so
    releasing just above that same height puts it back down instead of dropping it.
    """
    await place_at(m, x, y, grasp_z + m.workspace["pick"]["release_clearance"])


async def static_cycle(m: Manipulator) -> bool:
    """Blind pick -> place between the two fixed poses in poses.yaml. No camera involved."""
    s = m.poses["static"]
    offset = m.workspace["gripper"]["tcp_offset"]
    # Static poses are stored as gripper-frame z (as proven in the original
    # move_arm.py), so the pose commanded is identical whatever tcp_offset is.
    pick_pose, place_pose = s["pick"], s["place"]
    if not await pick_at(m, pick_pose["x"], pick_pose["y"], pick_pose["frame_z"] - offset, pick_pose["theta"]):
        return False
    await place_at(m, place_pose["x"], place_pose["y"], place_pose["frame_z"] - offset, place_pose["theta"])
    return True
