import asyncio
from viam.robot.client import RobotClient
import os
from dotenv import load_dotenv
from typing import ClassVar, Mapping, Optional, Sequence, Tuple, cast

from typing_extensions import Self
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic as GenericService
from viam.services.motion import MotionClient
from viam.utils import ValueTypes, struct_to_dict

load_dotenv()
api_key = os.getenv("API_KEY")
api_key_id = os.getenv("API_KEY_ID")
robot_address = os.getenv("ROBOT_ADDRESS")

async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=api_key,
        api_key_id=api_key_id
    )
    return await RobotClient.at_address(robot_address, opts)

# --- Default dependency names -------------------------------------------------
# Resource names are user-chosen. Override through the module resource's
# attributes ("gripper") if your gripper is named differently; the motion
# service's name is fixed by the platform.
DEFAULT_GRIPPER_NAME = "gripper"

# --- Defined poses (see the assisted setup's step 3.4 pick style) ----------------
# Gripper frame origin poses in the world frame. Side grasp (default): gripper axis
# parallel to the table. Top-down alternative: see references/workspace-poses.md.
# If your gripper's fingers open top-to-bottom instead, set theta=90 in both (side only).
PICK_POSE = Pose(x=300, y=-150, z=75, o_x=0, o_y=0, o_z=-1, theta=0)
PLACE_POSE = Pose(x=300, y=150, z=75, o_x=0, o_y=0, o_z=-1, theta=0)

APPROACH_MM = 100  # standoff above a pose before descending onto it
LIFT_MM = 150  # carrying height above a pose after grasping


def offset_pose(pose: Pose, z_offset_mm: float) -> Pose:
    """Raise or lower a pose in z while keeping x/y/orientation fixed."""
    return Pose(
        x=pose.x,
        y=pose.y,
        z=pose.z + z_offset_mm,
        o_x=pose.o_x,
        o_y=pose.o_y,
        o_z=pose.o_z,
        theta=pose.theta,
    )


# IMPORTANT: keep the class name and MODEL line your inline editor generated.
# The platform uses those auto-generated values to identify your module, and
# changing them breaks it. The ModelFamily below is a placeholder.
class MyGenericService(GenericService, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("hackathons", "8ce08475-ece9-436b-acd5-2d213944fb83"),
        "generic-service",
    )

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        # Required dependencies: viam-server holds this module back until they
        # are online. The gripper's name comes from the module's attributes;
        # the motion service is named by its full, platform-fixed resource name.
        required_deps = [
            attrs.get("gripper", DEFAULT_GRIPPER_NAME),
            "rdk:service:motion/builtin",
        ]
        return required_deps, []

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        self = super().new(config, dependencies)
        attrs = struct_to_dict(config.attributes)
        self.gripper_name = attrs.get("gripper", DEFAULT_GRIPPER_NAME)
        self.gripper = cast(
            Gripper, dependencies[Gripper.get_resource_name(self.gripper_name)]
        )
        self.motion = cast(
            MotionClient, dependencies[MotionClient.get_resource_name("builtin")]
        )
        return self

    async def move_gripper_to(self, pose: Pose) -> None:
        """Plan and execute a collision-free path putting the gripper TCP at pose."""
        await self.motion.move(
            component_name=self.gripper_name,
            destination=PoseInFrame(reference_frame="world", pose=pose),
        )

    async def open_gripper(self) -> None:
        """Open the gripper. UFactory Open waits for the jaw move to finish."""
        await self.gripper.open()

    async def grab(self, *, expect_holding: bool) -> bool:
        """Close the gripper. UFactory Grab waits for the jaw move and returns success.

        Same pattern as viam-pouring-demo GrabCup: await Grab and use its bool.
        Do not poll is_moving afterward — the driver only reports moving during the
        Grab RPC, so a post-await is_moving check is always false and is not a
        settle signal. expect_holding True requires Grab's bool; False (empty-air
        demos like go-to-pick) only waits for Grab to complete.
        """
        got = await self.gripper.grab()
        if not expect_holding:
            return True
        return bool(got)

    async def go_to_pick(self) -> None:
        """Open, approach from above, descend, grasp, then raise clear."""
        await self.open_gripper()
        await self.move_gripper_to(offset_pose(PICK_POSE, APPROACH_MM))
        await self.move_gripper_to(PICK_POSE)
        await self.grab(expect_holding=False)
        # Raise before any later lateral move so the object does not slide.
        await self.move_gripper_to(offset_pose(PICK_POSE, LIFT_MM))

    async def go_to_place(self) -> None:
        """Approach the place pose from above, open, and lift clear."""
        await self.move_gripper_to(offset_pose(PLACE_POSE, LIFT_MM))
        await self.move_gripper_to(PLACE_POSE)
        await self.open_gripper()
        await self.move_gripper_to(offset_pose(PLACE_POSE, LIFT_MM))

    async def run_static_cycle(self) -> bool:
        """Move the block from the pick point to the place point."""
        await self.open_gripper()
        await self.move_gripper_to(offset_pose(PICK_POSE, APPROACH_MM))
        await self.move_gripper_to(PICK_POSE)
        if not await self.grab(expect_holding=True):
            return False
        # Raise clear of the table before any lateral move to place.
        await self.move_gripper_to(offset_pose(PICK_POSE, LIFT_MM))
        await self.move_gripper_to(offset_pose(PLACE_POSE, LIFT_MM))
        await self.go_to_place()
        return True

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        action = command.get("action")
        if action == "static-cycle":
            return {"success": await self.run_static_cycle()}
        if action == "go-to-pick":
            await self.go_to_pick()
            return {"success": True}
        if action == "go-to-place":
            await self.go_to_place()
            return {"success": True}
        return {"error": "unknown command"}
async def main() -> None:
    robot = await connect()
    try:
        print("Resources:", [r.name for r in robot.resource_names])

        # Build the service locally, wiring its dependencies from the robot
        # client instead of from viam-server's module dependency map.
        service = MyGenericService("local-generic-service")
        service.gripper_name = DEFAULT_GRIPPER_NAME
        service.gripper = Gripper.from_robot(robot, DEFAULT_GRIPPER_NAME)
        service.motion = MotionClient.from_robot(robot, "builtin")

        result = await service.do_command({"action": "static-cycle"})
        print("Result:", result)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
