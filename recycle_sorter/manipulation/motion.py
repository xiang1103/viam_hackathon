from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient

from .safety import UnsafeTarget, check_frame_pose, check_target

log = logging.getLogger(__name__)


class Manipulator:
    """Thin, safety-checked wrapper over arm + gripper + motion service.

    Every arm motion goes through the motion planner (never raw joint commands), so
    the workcell obstacles in the machine config are respected on every move.

    dry_run: log every action, execute nothing.
    step:    wait for Enter before every motion.
    """

    def __init__(
        self,
        arm: Arm | None,
        gripper: Gripper,
        motion: MotionClient,
        machine_cfg: dict[str, Any],
        workspace: dict[str, Any],
        poses: dict[str, Any],
        dry_run: bool = False,
        step: bool = False,
    ):
        self.arm, self.gripper, self.motion = arm, gripper, motion
        self.workspace, self.poses = workspace, poses
        self.move_frame = machine_cfg["move_frame"]
        self.timeout = machine_cfg["rpc_timeout_s"]
        self.trust_is_holding = machine_cfg.get("trust_is_holding", False)
        self.normal_speed = machine_cfg.get("arm_speed")
        self.dry_run, self.step = dry_run, step

    async def _confirm(self, what: str) -> None:
        log.info("%s%s", "[dry-run] " if self.dry_run else "", what)
        if self.step and not self.dry_run:
            await asyncio.to_thread(input, f"  ENTER to: {what} (Ctrl-C aborts) ")

    async def _move(self, pose: Pose, linear: bool = False) -> None:
        dest = PoseInFrame(reference_frame="world", pose=pose)
        if linear:
            tol = self.workspace["pick"]["line_tolerance_mm"]
            constraints = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=tol)])
            try:
                if await self.motion.move(self.move_frame, dest, constraints=constraints, timeout=self.timeout):
                    return
            except Exception as e:
                # A failed plan has not moved the arm, so an unconstrained retry
                # is safe. Free plans are what move_arm.py proved on hardware.
                log.warning("linear plan failed (%s); retrying unconstrained", e)
        if not await self.motion.move(self.move_frame, dest, timeout=self.timeout):
            raise RuntimeError("motion.move returned False")

    async def move_to(self, x: float, y: float, z: float, theta: float = 0.0, linear: bool = False) -> None:
        """Put the FINGERTIPS at (x, y, z) in the world frame, gripper pointing straight down.

        Bounds are checked on the fingertip position; the pose sent to the planner is
        the gripper frame origin, tcp_offset above it, which has its own floor.
        """
        check_target(x, y, z, self.workspace)
        frame_z = z + self.workspace["gripper"]["tcp_offset"]
        floor = self.workspace["bounds"]["frame_z_min"]
        if frame_z < floor:
            raise UnsafeTarget(f"gripper frame z={frame_z:.0f} is below the {floor} mm floor (wrist would hit the table)")
        await self._confirm(f"move {'linear ' if linear else ''}to x={x:.0f} y={y:.0f} z={z:.0f} theta={theta:.0f}")
        if not self.dry_run:
            await self._move(Pose(x=x, y=y, z=frame_z, o_x=0, o_y=0, o_z=-1, theta=theta), linear)

    async def goto_named(self, name: str) -> None:
        """Planner move to a stored gripper-frame pose (poses.yaml -> named)."""
        named = self.poses.get("named", {})
        if name not in named and name == "home":
            name = "survey"
        if name not in named:
            raise KeyError(f"pose {name!r} not taught yet - run scripts/01_teach_pose.py {name}")
        p = named[name]
        check_frame_pose(p["x"], p["y"], p["z"], self.workspace)
        await self._confirm(f"move to '{name}' pose")
        if self.dry_run:
            return
        await self._move(Pose(**{k: float(p[k]) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}))
        await asyncio.sleep(0.3)  # let the wrist camera settle before a capture

    async def set_speed(self, degs_per_sec: float | None) -> None:
        """Arm joint speed. The xArm driver ignores the motion service's speed, so this is the only lever."""
        if self.dry_run or self.arm is None or not degs_per_sec:
            return
        await self.arm.do_command({"set_speed": float(degs_per_sec)})

    @asynccontextmanager
    async def slow(self):
        """Gentle joint speed for the final run-in onto an object; always restored afterwards."""
        await self.set_speed(self.workspace["pick"].get("grasp_speed"))
        try:
            yield
        finally:
            await self.set_speed(self.normal_speed)

    async def open(self) -> None:
        await self._confirm("open gripper")
        if not self.dry_run:
            await self.gripper.open(timeout=self.timeout)

    async def grab(self) -> bool:
        await self._confirm("grab")
        if self.dry_run:
            return True
        # UFactory grab() blocks until the jaws finish and returns whether something
        # is held. Do not poll is_moving afterwards: it is always false by then.
        grabbed = await self.gripper.grab(timeout=self.timeout)
        if not self.trust_is_holding:
            return grabbed
        status = await self.gripper.is_holding_something(timeout=self.timeout)
        return bool(status.is_holding_something)

    async def stop(self) -> None:
        if not self.dry_run and self.arm is not None:
            await self.arm.stop()
