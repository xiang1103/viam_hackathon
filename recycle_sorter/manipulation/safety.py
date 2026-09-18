from __future__ import annotations

from typing import Any


class UnsafeTarget(Exception):
    pass


def check_target(x: float, y: float, z: float, workspace: dict[str, Any]) -> None:
    """Reject any gripper target outside the configured workspace, before it reaches the planner."""
    b = workspace["bounds"]
    z_min = workspace["table_top"] + b["z_min_above_table"]
    problems = []
    if not b["x"][0] <= x <= b["x"][1]:
        problems.append(f"x={x:.0f} outside {b['x']}")
    if not b["y"][0] <= y <= b["y"][1]:
        problems.append(f"y={y:.0f} outside {b['y']}")
    if not z_min <= z <= b["z_max"]:
        problems.append(f"z={z:.0f} outside [{z_min:.0f}, {b['z_max']}]")
    if problems:
        raise UnsafeTarget("; ".join(problems))


def check_frame_pose(x: float, y: float, z: float, workspace: dict[str, Any]) -> None:
    """Bounds for a pose given directly as the gripper FRAME origin (named poses such as survey)."""
    b = workspace["bounds"]
    x_range = [b.get("named_x_min", b["x"][0]), b["x"][1]]
    problems = []
    if not x_range[0] <= x <= x_range[1]:
        problems.append(f"x={x:.0f} outside {x_range}")
    if not b["y"][0] <= y <= b["y"][1]:
        problems.append(f"y={y:.0f} outside {b['y']}")
    if not b["frame_z_min"] <= z <= b["frame_z_max"]:
        problems.append(f"frame z={z:.0f} outside [{b['frame_z_min']}, {b['frame_z_max']}]")
    if problems:
        raise UnsafeTarget("; ".join(problems))
