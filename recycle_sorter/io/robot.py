from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.media.video import CameraMimeType
from viam.proto.common import Pose, PoseInFrame, Transform
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from ..config import load_yaml, viam_credentials
from ..manipulation.motion import Manipulator
from ..perception import viam_vision
from ..perception.frames import cam_to_world_matrix
from ..perception.segment import segment
from ..types import Frame, Intrinsics, ObjectObservation

log = logging.getLogger(__name__)


async def connect() -> RobotClient:
    address, key_id, key = viam_credentials()
    opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    return await RobotClient.at_address(address, opts)


def find_camera_name(machine: RobotClient, configured: str | None) -> str:
    if configured:
        return configured
    cameras = [r.name for r in machine.resource_names if r.subtype == "camera"]
    if len(cameras) != 1:
        raise SystemExit(f"Found cameras {cameras}; set `camera:` in config/machine.yaml.")
    return cameras[0]


class LiveRobot:
    def __init__(self, machine: RobotClient, cfg: dict[str, Any], manipulator: Manipulator):
        self.machine, self.cfg, self.manip = machine, cfg, manipulator
        self._camera: Camera | None = None
        self._intrinsics: Intrinsics | None = None
        self._vision: dict[str, VisionClient] | None = None

    # Resolved on first use, so arm-only actions work on a machine with no camera.
    @property
    def camera_name(self) -> str:
        return find_camera_name(self.machine, self.cfg.get("camera"))

    @property
    def camera(self) -> Camera:
        if self._camera is None:
            self._camera = Camera.from_robot(self.machine, self.camera_name)
        return self._camera

    @classmethod
    async def create(cls, dry_run: bool = False, step: bool = False) -> "LiveRobot":
        cfg = load_yaml("machine.yaml")
        machine = await connect()
        arm = Arm.from_robot(machine, cfg["arm"])
        manip = Manipulator(
            arm,
            Gripper.from_robot(machine, cfg["gripper"]),
            MotionClient.from_robot(machine, cfg["motion"]),
            cfg,
            load_yaml("workspace.yaml"),
            load_yaml("poses.yaml"),
            dry_run=dry_run,
            step=step,
        )
        await manip.set_speed(cfg.get("arm_speed"))
        return cls(machine, cfg, manip)

    async def intrinsics(self) -> Intrinsics:
        if self._intrinsics is None:
            p = (await self.camera.get_properties()).intrinsic_parameters
            self._intrinsics = Intrinsics(
                p.width_px, p.height_px, p.focal_x_px, p.focal_y_px, p.center_x_px, p.center_y_px
            )
        return self._intrinsics

    async def _transform_point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """A camera-frame point in the world frame.

        The point is registered as a temporary frame hanging off the camera and the
        motion service is asked where that frame is. Unlike RobotClient.transform_pose
        this also works inside a module (technique proven on the arm in move_arm.py).
        """
        probe = Transform(
            reference_frame="sorter_probe",
            pose_in_observer_frame=PoseInFrame(reference_frame=self.camera_name, pose=Pose(x=x, y=y, z=z, o_z=1)),
        )
        p = (await self.manip.motion.get_pose("sorter_probe", "world", [probe], timeout=self.cfg["rpc_timeout_s"])).pose
        return p.x, p.y, p.z

    async def snapshot(self) -> Frame:
        images, _ = await self.camera.get_images(timeout=self.cfg["rpc_timeout_s"])
        color = depth = None
        for img in images:
            if img.mime_type == CameraMimeType.VIAM_RAW_DEPTH:
                depth = np.array(img.bytes_to_depth_array(), dtype=np.uint16)
            elif "depth" in img.name.lower():
                depth = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.uint16)
            elif color is None:
                color = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
        if color is None or depth is None:
            got = [(i.name, str(i.mime_type)) for i in images]
            raise RuntimeError(f"need a color and a depth image from {self.camera_name}, got {got}")

        if depth.shape != color.shape[:2]:
            # Different sizes means the streams are not pixel-aligned. Resizing is
            # only a stopgap: set align_color_depth on the RealSense for real use.
            log.warning("depth %s != color %s: enable align_color_depth on the camera", depth.shape, color.shape[:2])
            depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        joints = await self.manip.arm.get_joint_positions()  # read-only: recorded with the frame
        return Frame(
            color=color,
            depth=depth,
            intrinsics=await self.intrinsics(),
            cam_to_world=await cam_to_world_matrix(self._transform_point),
            joints=list(joints.values),
            timestamp=datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        )

    async def detect(self, frame: Frame) -> list[ObjectObservation]:
        """Objects in the unsorted zone, from Viam vision or the OpenCV fallback (machine.yaml -> perception)."""
        if self.cfg.get("perception", "viam") != "viam":
            return segment(frame, self.manip.workspace)
        vision = self.cfg["vision"]
        if self._vision is None:
            self._vision = {label: VisionClient.from_robot(self.machine, name) for label, name in vision["segmenters"].items()}
        return await viam_vision.detect(
            self._vision,
            self.camera_name,
            frame,
            self.manip.workspace,
            use_detection_label=vision.get("use_detection_label", False),
            timeout=self.cfg["rpc_timeout_s"],
        )

    async def close(self) -> None:
        await self.machine.close()
