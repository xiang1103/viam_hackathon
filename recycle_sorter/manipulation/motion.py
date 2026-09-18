from __future__ import annotations

import asyncio
import logging
from typing import Any

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.component.arm import JointPositions
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient

from .safety import check_target

log = logging.getLogger(__name__)


class Manipulator:
    """Thin, safety-checked wrapper over arm + gripper + motion service.

    dry_run: log every action, execute nothing.
    step:    wait for Enter before every motion.
    """

    def __init__(
        self,
        arm: Arm,
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
        self.dry_run, self.step = dry_run, step

    async def _confirm(self, what: str) -> None:
        log.info("%s%s", "[dry-run] " if self.dry_run else "", what)
        if self.step and not self.dry_run:
            await asyncio.to_thread(input, f"  ENTER to: {what} (Ctrl-C aborts) ")

    async def move_to(self, x: float, y: float, z: float, theta: float = 0.0, linear: bool = False) -> None:
        """Put the FINGERTIPS at (x, y, z) in the world frame, gripper pointing straight down.

        Bounds are checked on the fingertip position; the pose sent to the planner
        is the gripper frame origin, tcp_offset above it.
        """
        check_target(x, y, z, self.workspace)
        await self._confirm(f"move {'linear ' if linear else ''}to x={x:.0f} y={y:.0f} z={z:.0f} theta={theta:.0f}")
        if self.dry_run:
            return
        frame_z = z + self.workspace["gripper"]["tcp_offset"]
        dest = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=frame_z, o_x=0, o_y=0, o_z=-1, theta=theta))
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

    async def goto_named(self, name: str) -> None:
        """Joint-space move to a taught pose: repeatable, and needs no planner."""
        if name not in self.poses.get("joints", {}):
            raise KeyError(f"pose {name!r} not taught yet - run scripts/01_teach_pose.py {name}")
        await self._confirm(f"joint move to '{name}'")
        if self.dry_run:
            return
        await self.arm.move_to_joint_positions(JointPositions(values=self.poses["joints"][name]), timeout=self.timeout)
        while await self.arm.is_moving():
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)  # let the wrist camera settle before a capture

    async def open(self) -> None:
        await self._confirm("open gripper")
        if not self.dry_run:
            await self.gripper.open(timeout=self.timeout)

    async def grab(self) -> bool:
        await self._confirm("grab")
        if self.dry_run:
            return True
        grabbed = await self.gripper.grab(timeout=self.timeout)
        # is_holding_something is only reliable on some grippers (e.g. xArm G2).
        # Enable trust_is_holding in machine.yaml once verified on the real one.
        if not self.trust_is_holding:
            return grabbed
        status = await self.gripper.is_holding_something(timeout=self.timeout)
        return bool(status.is_holding_something)

    async def stop(self) -> None:
        if not self.dry_run:
            await self.arm.stop()
