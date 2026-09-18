"""This package's motions behind the same do_command interface as the hackathon module.

The team's deployed module code is move_arm.py; this mirrors its class name, MODEL and
action names so the package can be dropped into the module later without renaming anything.

Runs two ways:
  * in viam-server, as the module: dependencies arrive through `new()`.
  * from a laptop: `run_local()` wires the same class from a RobotClient.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Mapping, Optional, Sequence, Tuple, cast

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic as GenericService
from viam.services.motion import MotionClient
from viam.utils import ValueTypes, struct_to_dict

from .config import load_yaml
from .manipulation.motion import Manipulator
from .manipulation.pickplace import pick_at, place_at, static_cycle

if TYPE_CHECKING:
    from .io.robot import LiveRobot

DEFAULT_ARM_NAME = "arm"
DEFAULT_GRIPPER_NAME = "gripper"


# IMPORTANT: keep the class name and MODEL line the inline editor generated.
# The platform uses those auto-generated values to identify the module, and
# changing them breaks it.
class MyGenericService(GenericService, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("hackathons", "8ce08475-ece9-436b-acd5-2d213944fb83"),
        "generic-service",
    )

    manip: Manipulator
    live: Optional["LiveRobot"] = None  # set only by run_local(); needed for camera-driven actions

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        # Required dependencies: viam-server holds this module back until they are
        # online. The motion service is named by its platform-fixed resource name.
        required_deps = [
            attrs.get("arm", DEFAULT_ARM_NAME),
            attrs.get("gripper", DEFAULT_GRIPPER_NAME),
            "rdk:service:motion/builtin",
        ]
        return required_deps, []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        attrs = struct_to_dict(config.attributes)
        arm_name = attrs.get("arm", DEFAULT_ARM_NAME)
        gripper_name = attrs.get("gripper", DEFAULT_GRIPPER_NAME)
        # The frame motion.move drives is the gripper, whatever it is named here.
        machine_cfg = {**load_yaml("machine.yaml"), "move_frame": gripper_name}
        self.manip = Manipulator(
            cast(Arm, dependencies[Arm.get_resource_name(arm_name)]),
            cast(Gripper, dependencies[Gripper.get_resource_name(gripper_name)]),
            cast(MotionClient, dependencies[MotionClient.get_resource_name("builtin")]),
            machine_cfg,
            load_yaml("workspace.yaml"),
            load_yaml("poses.yaml"),
        )
        return self

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        action = command.get("action")
        m = self.manip
        offset = m.workspace["gripper"]["tcp_offset"]

        if action == "static-cycle":
            return {"success": await static_cycle(m)}
        if action == "go-to-pick":
            s = m.poses["static"]["pick"]
            await pick_at(m, s["x"], s["y"], s["frame_z"] - offset, s["theta"])  # empty-air demo: result ignored
            return {"success": True}
        if action == "go-to-place":
            s = m.poses["static"]["place"]
            await place_at(m, s["x"], s["y"], s["frame_z"] - offset, s["theta"])
            return {"success": True}
        if action == "sort":
            if self.live is None:
                # Not wired for module mode yet: the camera would have to arrive as a
                # dependency. Everything else already works without a RobotClient.
                return {"error": "sort is only wired for local runs so far: python -m recycle_sorter.cli"}
            from .app import run_sort

            counts = await run_sort(self.live, str(command.get("mode", "color")), int(command.get("max_picks", 50)))
            return {"success": True, "sorted": dict(counts)}
        return {"error": "unknown command"}


async def run_local(command: Mapping[str, ValueTypes], dry_run: bool = False, step: bool = False) -> Mapping[str, ValueTypes]:
    """Drive the service from a laptop, wiring its dependencies from a RobotClient."""
    from .io.robot import LiveRobot

    robot = await LiveRobot.create(dry_run=dry_run, step=step)
    try:
        print("Resources:", [r.name for r in robot.machine.resource_names])
        service = MyGenericService("local-generic-service")
        service.manip, service.live = robot.manip, robot
        return await service.do_command(command)
    except BaseException:
        await robot.manip.stop()
        raise
    finally:
        await robot.close()
