from __future__ import annotations

import logging

from ..types import ObjectObservation
from .motion import Manipulator

log = logging.getLogger(__name__)


def grasp_pose(obs: ObjectObservation, workspace: dict) -> tuple[float, float, float, float]:
    """Top-down grasp: (x, y, z, theta). Fingers close across the object's short axis."""
    g = workspace["gripper"]
    z = max(obs.top_z - g["finger_depth"], workspace["table_top"] + workspace["bounds"]["z_min_above_table"])
    return float(obs.centroid[0]), float(obs.centroid[1]), z, obs.yaw_deg + g["yaw_offset_deg"]


async def pick(m: Manipulator, obs: ObjectObservation) -> bool:
    """Returns True if the gripper reports holding something after the lift."""
    p = m.workspace["pick"]
    x, y, z, theta = grasp_pose(obs, m.workspace)
    above = obs.top_z + p["pregrasp_above"]

    await m.open()
    try:
        await m.move_to(x, y, above, theta)
    except Exception as e:
        # A 180 degree flip is the same grasp for a parallel gripper, but can be
        # reachable when the first wrist angle is not.
        log.warning("pre-grasp failed (%s); retrying with theta+180", e)
        theta += 180.0
        await m.move_to(x, y, above, theta)
    await m.move_to(x, y, z, theta, linear=True)
    held = await m.grab()
    await m.move_to(x, y, obs.top_z + p["lift"], theta, linear=True)
    return held


async def place(m: Manipulator, bin_name: str) -> None:
    bins = m.poses.get("bins", {})
    if bin_name not in bins:
        raise KeyError(f"bin {bin_name!r} not taught yet - run scripts/01_teach_pose.py --bin {bin_name}")
    p = m.workspace["pick"]
    x, y = bins[bin_name]["x"], bins[bin_name]["y"]
    release_z = m.workspace["table_top"] + p["place_above"]

    await m.move_to(x, y, release_z + p["lift"])
    await m.move_to(x, y, release_z, linear=True)
    await m.open()
    await m.move_to(x, y, release_z + p["lift"], linear=True)
