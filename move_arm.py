from connect import connect
import asyncio
import math
from typing import ClassVar, Mapping, Optional, Sequence, Tuple, cast

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName, Transform
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic as GenericService
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.utils import ValueTypes, struct_to_dict

# --- Default dependency names -------------------------------------------------
# Resource names are user-chosen. Override through the module resource's
# attributes ("gripper", "segmenter", "camera", "arm") if yours are named
# differently; the motion service's name is fixed by the platform.
DEFAULT_GRIPPER_NAME = "gripper"
DEFAULT_SEGMENTER_NAME = "object-segmenter"
DEFAULT_CAMERA_NAME = "cam"
DEFAULT_ARM_NAME = "arm"

# --- Defined poses (see the assisted setup's step 3.4 pick style) ----------------
# Gripper frame origin poses in the world frame. Side grasp (default): gripper axis
# parallel to the table. Top-down alternative: see references/workspace-poses.md.
# If your gripper's fingers open top-to-bottom instead, set theta=90 in both (side only).
PICK_POSE = Pose(x=300, y=-150, z=75, o_x=0, o_y=0, o_z=-1, theta=0)
PLACE_POSE = Pose(x=300, y=150, z=75, o_x=0, o_y=0, o_z=-1, theta=0)
WATCH_POSE = Pose(x=197.45970981685434, y=0.1396234792825357, z=616.4408111332061, o_x=0.3746224611726016, o_y=0.0007932389645242312, o_z=-0.9271771040944273, theta=-4.119648697216458)

# Grasp geometry for a detected object. go-to-detected follows PICK_POSE:
# top-down when abs(PICK_POSE.o_z) > 0.5 (gripper axis down, TCP raised by
# FINGERTIP_OFFSET_MM above the grasp-band center, vertical approach/retreat),
# otherwise a HORIZONTAL grasp facing the object from the arm base. Both
# styles clamp TCP z to a table-clearance floor — side below ~80 mm drives
# gripper-1:case-gripper into the table (case ~100 mm tall, half below TCP);
# top-down TCP below ~60 mm drives wrist links into the table (see
# references/workspace-poses.md "Minimum gripper Z"). When pick style is side
# but the detected center is below the side floor, auto-switch THAT grasp to
# top-down for IK clearance — and detected-cycle must place with the same
# style (mirror pick → place). Placing with side PLACE_POSE after a top-down
# auto-pick rotates the object ~90°. Keep PICK_POSE / PLACE_POSE in the same
# pick style as the base cycle (static cycle is unchanged by the auto-switch).
#
# OBJECT_HEIGHT_MM: standing height above the table (mm) when the user chose
# top-down. Top-down grasps use height/2 as the grasp-band center instead of
# the point-cloud center — guessing from detection underestimates tall objects
# and crashes the gripper into them. None for side style.
GRASP_THETA = 0.0
FINGERTIP_OFFSET_MM = 60  # TCP above object center for top-down fingertip reach
MIN_SIDE_GRASP_Z_MM = 80  # floor for horizontal TCP / object-center height
MIN_TOP_DOWN_TCP_Z_MM = 60  # floor for top-down TCP above the table
OBJECT_HEIGHT_MM: Optional[float] = 30.0  # set for top-down; standing height mm

MAX_OBJECT_DIM_MM = 150  # larger segments are noise, not a block
MAX_PICK_REACH_MM = 700  # detections farther from the arm base (xy) are rejected

APPROACH_MM = 150  # standoff before the grasp (above for top-down, in front for side)
LIFT_MM = 150  # carrying distance backed off after grasping (horizontal for side)
LIFT_UP_MM = 150  # vertical clearance raised after grasping so the object clears the table
# The xArm driver executes motion at its own joint speed (speed_degs_per_sec),
# which the builtin motion service's linear speed does NOT override. So we set
# the arm's speed directly via its set_speed DoCommand for the grasp run-in and
# restore it afterward. Values are joint deg/sec (arm accepts 3-180).
GRASP_SPEED_DEGS_PER_SEC = 8.0  # slow, gentle joint speed for the final run-in
NORMAL_SPEED_DEGS_PER_SEC = 60.0  # restored after the grasp (driver default)


def is_top_down_pick() -> bool:
    """True when PICK_POSE is a top-down grasp (gripper axis points down)."""
    return abs(PICK_POSE.o_z) > 0.5


def should_grasp_top_down(object_z: float) -> bool:
    """Top-down when PICK_POSE is top-down, or when a side grasp would be too low.

    A horizontal approach with TCP below MIN_SIDE_GRASP_Z_MM puts
    case-gripper into the table — common for short blocks (center z ≈ 15–40).
    Those detections use a top-down grasp even if the fixed poses are side style.
    detected-cycle must place with the same style (see place_pose_matching_grasp).
    """
    if is_top_down_pick():
        return True
    return object_z < MIN_SIDE_GRASP_Z_MM


def grasp_pose_for_object(x: float, y: float, z: float) -> Pose:
    """Build a grasp pose at the detected object using pick style + Z floors.

    Top-down (PICK_POSE or object below the side floor): TCP at
    max(center_z + FINGERTIP_OFFSET_MM, MIN_TOP_DOWN_TCP_Z_MM) with o_z = -1.
    When the user chose top-down and OBJECT_HEIGHT_MM is set, center_z is
    height/2 (not the point-cloud center) so tall objects are not underestimated.
    Side auto-switches to top-down still use the detected z. Side: horizontal
    pose at (x, y, z) with TCP z at least MIN_SIDE_GRASP_Z_MM.
    See references/workspace-poses.md "Minimum gripper Z".
    """
    if should_grasp_top_down(z):
        if is_top_down_pick() and OBJECT_HEIGHT_MM is not None:
            center_z = OBJECT_HEIGHT_MM / 2.0
        else:
            center_z = z
        return Pose(
            x=x,
            y=y,
            z=max(center_z + FINGERTIP_OFFSET_MM, MIN_TOP_DOWN_TCP_Z_MM),
            o_x=0.0,
            o_y=0.0,
            o_z=-1.0,
            theta=GRASP_THETA,
        )
    heading = math.hypot(x, y)
    if heading == 0.0:
        ox, oy = 1.0, 0.0
    else:
        ox, oy = x / heading, y / heading
    return Pose(
        x=x,
        y=y,
        z=max(z, MIN_SIDE_GRASP_Z_MM),
        o_x=ox,
        o_y=oy,
        o_z=0.0,
        theta=GRASP_THETA,
    )


def place_pose_matching_grasp(grasp: Pose, place_z: float) -> Pose:
    """PLACE_POSE x/y at place_z, orientation matching the actual grasp style.

    Mirrors pick → place so an auto top-down pick is not followed by a side
    PLACE_POSE drop (which rotates the object ~90°). Side place keeps the
    designed PLACE_POSE heading at the place marker; top-down place uses o_z=-1.
    """
    if abs(grasp.o_z) > 0.5:
        return Pose(
            x=PLACE_POSE.x,
            y=PLACE_POSE.y,
            z=place_z,
            o_x=0.0,
            o_y=0.0,
            o_z=-1.0,
            theta=GRASP_THETA,
        )
    return Pose(
        x=PLACE_POSE.x,
        y=PLACE_POSE.y,
        z=place_z,
        o_x=PLACE_POSE.o_x,
        o_y=PLACE_POSE.o_y,
        o_z=PLACE_POSE.o_z,
        theta=PLACE_POSE.theta,
    )


def horizontal_offset_pose(pose: Pose, dist_mm: float) -> Pose:
    """Offset a pose in the world XY plane along its own gripper heading.

    Keeps z and orientation fixed; moves only in x/y so the approach and
    retreat stay horizontal. Positive dist_mm backs off opposite the heading.
    """
    return Pose(
        x=pose.x - dist_mm * pose.o_x,
        y=pose.y - dist_mm * pose.o_y,
        z=pose.z,
        o_x=pose.o_x,
        o_y=pose.o_y,
        o_z=pose.o_z,
        theta=pose.theta,
    )


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
        # are online. The vision segmenter and camera back the detection
        # actions; the motion service is named by its platform-fixed name.
        required_deps = [
            attrs.get("gripper", DEFAULT_GRIPPER_NAME),
            attrs.get("segmenter", DEFAULT_SEGMENTER_NAME),
            attrs.get("camera", DEFAULT_CAMERA_NAME),
            attrs.get("arm", DEFAULT_ARM_NAME),
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
        self.segmenter_name = attrs.get("segmenter", DEFAULT_SEGMENTER_NAME)
        self.camera_name = attrs.get("camera", DEFAULT_CAMERA_NAME)
        self.arm_name = attrs.get("arm", DEFAULT_ARM_NAME)
        # Z (mm) and pose at which the last object was grasped, reused for placing
        # (same set-down height) and for raising at the pick before a lateral move.
        self.last_pick_z: Optional[float] = None
        self.last_pick_pose: Optional[Pose] = None
        self.gripper = cast(
            Gripper, dependencies[Gripper.get_resource_name(self.gripper_name)]
        )
        self.motion = cast(
            MotionClient, dependencies[MotionClient.get_resource_name("builtin")]
        )
        self.segmenter = cast(
            VisionClient,
            dependencies[VisionClient.get_resource_name(self.segmenter_name)],
        )
        self.arm = cast(
            Arm, dependencies[Arm.get_resource_name(self.arm_name)]
        )
        return self

    async def move_gripper_to(self, pose: Pose) -> None:
        """Plan and execute a collision-free path putting the gripper TCP at pose (world frame).

        The grasp slow-down is done by setting the arm's joint speed via
        set_speed in go_to_detected, because the xArm driver ignores the motion
        service's linear speed.
        """
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

    async def go_to_watch(self) -> None:
        """Park at the watch pose with the gripper held open (camera aims at workspace)."""
        await self.open_gripper()
        await self.move_gripper_to(WATCH_POSE)

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

    async def closest_object_center(self) -> Optional[Pose]:
        """Return the closest detected object's center pose, in the camera frame.

        get_object_point_clouds runs the color detector + segmenter and returns
        one point-cloud object per detection, each with a bounding-box geometry
        whose center is expressed in the camera's frame. Of all detections we
        keep the one whose center is nearest the camera origin.
        """
        objects = await self.segmenter.get_object_point_clouds(self.camera_name)
        best_center: Optional[Pose] = None
        best_distance = float("inf")
        for obj in objects:
            geometries = obj.geometries.geometries
            if not geometries:
                continue
            geometry = geometries[0]
            # Skip segments far larger than a block (depth noise, table edges).
            if geometry.HasField("box") and max(
                geometry.box.dims_mm.x, geometry.box.dims_mm.y, geometry.box.dims_mm.z
            ) > MAX_OBJECT_DIM_MM:
                continue
            center = geometry.center
            distance = math.sqrt(center.x**2 + center.y**2 + center.z**2)
            if distance < best_distance:
                best_distance = distance
                best_center = center
        return best_center

    async def object_world_position(self) -> Optional[Tuple[float, float, float]]:
        """Detect the object and return its center as (x, y, z) in the world frame.

        closest_object_center returns the center in the CAMERA frame. An inline module has
        no RobotClient.transform_pose, so we register the detected point as a
        temporary frame attached to the camera and ask the motion service for
        that frame's pose in world via get_pose with a supplemental transform.
        """
        center = await self.closest_object_center()
        if center is None:
            return None
        supp = Transform(
            reference_frame="detected_object",
            pose_in_observer_frame=PoseInFrame(
                reference_frame=self.camera_name, pose=center
            ),
        )
        world_pif = await self.motion.get_pose(
            component_name="detected_object",
            destination_frame="world",
            supplemental_transforms=[supp],
        )
        p = world_pif.pose
        return (p.x, p.y, p.z)

    async def find_object(self) -> Mapping[str, ValueTypes]:
        """Report whether an object of the target color is visible, and where.

        Query only — the arm does not move. The position MUST be the object's
        center in the WORLD frame in mm (nested under world_position), the same
        frame PICK_POSE and PLACE_POSE use — never the camera-local center from
        get_object_point_clouds. Compare against the 3D scene's world position,
        not its local position under the camera.
        """
        pos = await self.object_world_position()
        if pos is None:
            return {"found": False}
        x, y, z = pos
        return {"found": True, "world_position": {"x": x, "y": y, "z": z}}

    async def go_to_detected(self) -> bool:
        """Detect the target object by color and grasp it.

        Uses PICK_POSE style, with a table-clearance Z floor. Top-down (or
        side when the detected center is below MIN_SIDE_GRASP_Z_MM): TCP
        above the center, approach and retreat straight up. Side only when the
        object is tall enough: horizontal grasp facing it from the arm base.
        Returns False when nothing of the target color is seen.
        """
        pos = await self.object_world_position()
        if pos is None:
            return False
        if math.hypot(pos[0], pos[1]) > MAX_PICK_REACH_MM or pos[2] < -20:
            print(f"Ignoring detection outside the workspace: {pos}")
            return False
        target = grasp_pose_for_object(*pos)
        self.last_pick_z = target.z
        self.last_pick_pose = target
        if should_grasp_top_down(pos[2]):
            await self.open_gripper()
            await self.move_gripper_to(offset_pose(target, APPROACH_MM))
            await self.arm.do_command({"set_speed": GRASP_SPEED_DEGS_PER_SEC})
            try:
                await self.move_gripper_to(target)
                if not await self.grab(expect_holding=True):
                    return False
            finally:
                await self.arm.do_command({"set_speed": NORMAL_SPEED_DEGS_PER_SEC})
            # Raise straight up before any later lateral move to place.
            await self.move_gripper_to(offset_pose(target, LIFT_UP_MM))
            return True
        await self.open_gripper()
        await self.move_gripper_to(horizontal_offset_pose(target, APPROACH_MM))
        # Slow the arm itself for the run-in, then restore normal speed.
        await self.arm.do_command({"set_speed": GRASP_SPEED_DEGS_PER_SEC})
        try:
            await self.move_gripper_to(target)
            if not await self.grab(expect_holding=True):
                return False
        finally:
            await self.arm.do_command({"set_speed": NORMAL_SPEED_DEGS_PER_SEC})
        # Back straight up first so the object clears the table, then retreat
        # horizontally away from where it was sitting.
        await self.move_gripper_to(offset_pose(target, LIFT_UP_MM))
        await self.move_gripper_to(horizontal_offset_pose(offset_pose(target, LIFT_UP_MM), LIFT_MM))
        return True

    async def run_detected_cycle(self) -> bool:
        """Pick the detected object, then place it at the fixed place x/y, set
        down at the same height it was picked from.

        Place orientation mirrors the actual grasp style (last_pick_pose), not
        blindly PLACE_POSE — so a short-object auto top-down pick is followed
        by a top-down place instead of a side drop that rotates the object.
        """
        # Detect from the watch pose: the wrist camera only sees the workspace
        # from there, and elsewhere it returns junk far outside reach.
        await self.go_to_watch()
        if not await self.go_to_detected():
            return False
        # Place at PLACE_POSE's x/y but at the grasped object's own height, so
        # it is set down gently instead of dropped from a fixed height.
        place_z = self.last_pick_z if self.last_pick_z is not None else PLACE_POSE.z
        if self.last_pick_pose is not None:
            place_at = place_pose_matching_grasp(self.last_pick_pose, place_z)
        else:
            place_at = Pose(
                x=PLACE_POSE.x,
                y=PLACE_POSE.y,
                z=place_z,
                o_x=PLACE_POSE.o_x,
                o_y=PLACE_POSE.o_y,
                o_z=PLACE_POSE.o_z,
                theta=PLACE_POSE.theta,
            )
        # Always raise at the pick first, then transit at carrying height — never
        # slide across the table at grasp z.
        if self.last_pick_pose is not None:
            await self.move_gripper_to(offset_pose(self.last_pick_pose, LIFT_UP_MM))
        await self.move_gripper_to(offset_pose(place_at, LIFT_UP_MM))
        await self.move_gripper_to(place_at)
        await self.open_gripper()
        await self.move_gripper_to(offset_pose(place_at, LIFT_UP_MM))
        # Return to the watch pose to await the next object.
        await self.go_to_watch()
        return True

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        action = command.get("action")

        # Full cycles: pick something up and put it down again.
        if action == "static-cycle":
            return {"success": await self.run_static_cycle()}
        if action == "detected-cycle":
            return {"success": await self.run_detected_cycle()}

        # Single moves: drive the arm to one place and do the one thing there.
        if action == "go-to-pick":
            await self.go_to_pick()
            return {"success": True}
        if action == "go-to-place":
            await self.go_to_place()
            return {"success": True}
        if action == "go-to-watch":
            await self.go_to_watch()
            return {"success": True}
        if action == "go-to-detected":
            return {"success": await self.go_to_detected()}

        # Query: reports what the camera sees; the arm does not move.
        if action == "find-object":
            return await self.find_object()

        return {"error": "unknown command"}

    

async def main() -> None:
    robot = await connect()
    try:
        print("Resources:", [r.name for r in robot.resource_names])

        # # Build the service locally, wiring its dependencies from the robot
        # # client instead of from viam-server's module dependency map.
        # service = MyGenericService("local-generic-service")
        # # Mirror everything new() sets, or methods hit AttributeError.
        # service.gripper_name = DEFAULT_GRIPPER_NAME
        # service.segmenter_name = DEFAULT_SEGMENTER_NAME
        # service.camera_name = DEFAULT_CAMERA_NAME
        # service.arm_name = DEFAULT_ARM_NAME
        # service.last_pick_z = None
        # service.last_pick_pose = None
        # service.gripper = Gripper.from_robot(robot, DEFAULT_GRIPPER_NAME)
        # service.motion = MotionClient.from_robot(robot, "builtin")
        # service.segmenter = VisionClient.from_robot(robot, DEFAULT_SEGMENTER_NAME)
        # service.arm = Arm.from_robot(robot, DEFAULT_ARM_NAME)

        # result = await service.do_command({"action": "detected-cycle"})
        # print("Result:", result)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
