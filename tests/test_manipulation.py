import pytest

from recycle_sorter.config import load_yaml
from recycle_sorter.io.recorder import load_frame, save_frame
from recycle_sorter.manipulation.motion import Manipulator
from recycle_sorter.manipulation.pickplace import grasp_pose, pick, place
from recycle_sorter.manipulation.safety import UnsafeTarget, check_target
from recycle_sorter.perception.segment import segment

from .synthetic import SCENE, TABLE_TOP, make_frame


@pytest.fixture
def workspace():
    ws = load_yaml("workspace.yaml")
    ws["table_top"] = TABLE_TOP
    ws["unsorted_zone"] = {"x": [230, 520], "y": [-150, 150]}
    return ws


def dry_manipulator(workspace, poses):
    cfg = {"move_frame": "gripper", "rpc_timeout_s": 5}
    return Manipulator(None, None, None, cfg, workspace, poses, dry_run=True)


def test_bounds_reject_targets_outside_workspace(workspace):
    ok_z = TABLE_TOP + 100
    check_target(400, 0, ok_z, workspace)
    off_the_table_edge = (400, 300, ok_z)  # measured: the table ends ~220-265 mm out on the +y side
    for bad in [(100, 0, ok_z), (720, 0, ok_z), (400, 480, ok_z), off_the_table_edge, (400, 0, TABLE_TOP + 2), (400, 0, 900)]:
        with pytest.raises(UnsafeTarget):
            check_target(*bad, workspace)


def test_grasp_never_goes_below_table_clearance(workspace):
    thin = [{"x": 400, "y": 0, "l": 40, "w": 40, "h": 10, "yaw": 0, "color": "red"}]
    obs = segment(make_frame(thin), workspace)[0]
    _, _, z, _ = grasp_pose(obs, workspace)
    assert z >= TABLE_TOP + workspace["bounds"]["z_min_above_table"]


def test_upright_round_item_gets_the_fixed_wrist_angle(workspace):
    workspace["gripper"].update(upright_theta=0, upright_min_height=80, yaw_offset_deg=0)
    can = [{"x": 400, "y": 0, "l": 66, "w": 66, "h": 122, "yaw": 37, "color": "red"}]
    block = [{"x": 400, "y": 0, "l": 60, "w": 30, "h": 30, "yaw": 37, "color": "red"}]
    can_obs, block_obs = (segment(make_frame(s), workspace)[0] for s in (can, block))
    assert grasp_pose(can_obs, workspace)[3] == 0  # whatever yaw was measured off its lid
    assert grasp_pose(block_obs, workspace)[3] == block_obs.yaw_deg != 0  # a block still has a short axis
    workspace["gripper"]["upright_theta"] = None
    assert grasp_pose(can_obs, workspace)[3] == can_obs.yaw_deg


async def test_dry_run_pick_and_place_sequence(workspace, caplog):
    caplog.set_level("INFO")
    obs = segment(make_frame(SCENE[:1]), workspace)[0]
    m = dry_manipulator(workspace, {})
    assert await pick(m, obs)
    await place(m, 300, -300, TABLE_TOP + 10)
    actions = [r.message.split("] ")[1].split(" to ")[0] for r in caplog.records]
    assert actions == [
        "open gripper", "move", "move linear", "grab", "move linear",  # pick
        "move", "move linear", "open gripper", "move linear",  # place
    ]


async def test_place_rejects_target_outside_workspace(workspace):
    with pytest.raises(UnsafeTarget):
        await place(dry_manipulator(workspace, {}), 900, 0, TABLE_TOP + 10)


def test_frame_round_trips_through_recorder(tmp_path, workspace):
    frame = make_frame(SCENE)
    loaded = load_frame(save_frame(frame, tmp_path))
    assert (loaded.depth == frame.depth).all() and (loaded.color == frame.color).all()
    assert len(segment(loaded, workspace)) == len(segment(frame, workspace))


# --- parity with the original move_arm.py ------------------------------------

class FakeMotion:
    def __init__(self, fail_linear=False):
        self.calls, self.fail_linear = [], fail_linear

    async def move(self, component_name, destination, world_state=None, constraints=None, timeout=None):
        if constraints is not None and self.fail_linear:
            raise RuntimeError("no path")
        p = destination.pose
        self.calls.append((component_name, destination.reference_frame, p.x, p.y, p.z, p.o_z, p.theta))
        return True


class FakeGripper:
    def __init__(self, holds=True):
        self.events, self.holds = [], holds

    async def open(self, timeout=None):
        self.events.append("open")

    async def grab(self, timeout=None):
        self.events.append("grab")
        return self.holds


def live_manipulator(workspace, motion, gripper):
    cfg = {"move_frame": "gripper", "rpc_timeout_s": 5}
    return Manipulator(None, gripper, motion, cfg, workspace, load_yaml("poses.yaml"))


# What move_arm.py's run_static_cycle sends to motion.move, minus its one redundant
# repeat of the place-lift pose: PICK/PLACE at z=75, approach +150, lift +150.
ORIGINAL_STATIC_CYCLE = [
    (300, -150, 225), (300, -150, 75), (300, -150, 225),
    (300, 150, 225), (300, 150, 75), (300, 150, 225),
]


@pytest.mark.parametrize("tcp_offset", [60, 70])
async def test_static_cycle_commands_the_original_poses(workspace, tcp_offset):
    from recycle_sorter.manipulation.pickplace import static_cycle

    workspace["gripper"]["tcp_offset"] = tcp_offset
    workspace["bounds"]["z_max"] = 600
    workspace["pick"].update(approach=150, lift=150)  # move_arm.py's APPROACH_MM / LIFT_MM
    motion, gripper = FakeMotion(), FakeGripper()
    assert await static_cycle(live_manipulator(workspace, motion, gripper))
    assert [(x, y, z) for _, _, x, y, z, _, _ in motion.calls] == ORIGINAL_STATIC_CYCLE
    assert all(c[0] == "gripper" and c[1] == "world" and c[5] == -1 for c in motion.calls)
    assert gripper.events == ["open", "grab", "open"]


async def test_static_cycle_stops_when_grab_reports_empty(workspace):
    from recycle_sorter.manipulation.pickplace import static_cycle

    motion = FakeMotion()
    assert not await static_cycle(live_manipulator(workspace, motion, FakeGripper(holds=False)))
    assert all(y == -150 for _, _, _, y, _, _, _ in motion.calls)  # never went to the place side


async def test_linear_descent_falls_back_to_free_plan(workspace):
    from recycle_sorter.manipulation.pickplace import static_cycle

    motion = FakeMotion(fail_linear=True)
    assert await static_cycle(live_manipulator(workspace, motion, FakeGripper()))
    assert len(motion.calls) == len(ORIGINAL_STATIC_CYCLE)


async def test_tcp_offset_lifts_the_commanded_frame_above_the_fingertips(workspace):
    workspace["gripper"]["tcp_offset"] = 70
    motion = FakeMotion()
    await live_manipulator(workspace, motion, FakeGripper()).move_to(400, 0, TABLE_TOP + 10)
    assert motion.calls[0][4] == TABLE_TOP + 80  # fingertips 10 above the table -> frame 80 above


async def test_gripper_frame_never_goes_below_the_hardware_floor(workspace):
    # move_arm.py: a top-down gripper frame below ~60 mm drives the wrist into the table.
    workspace["table_top"], workspace["gripper"]["tcp_offset"] = 0.0, 40
    motion = FakeMotion()
    with pytest.raises(UnsafeTarget, match="floor"):
        await live_manipulator(workspace, motion, FakeGripper()).move_to(400, 0, 10)  # frame would be 50
    assert motion.calls == []


class FakeArm:
    def __init__(self):
        self.speeds = []

    async def do_command(self, cmd):
        self.speeds.append(cmd["set_speed"])


async def test_descent_and_grab_run_slow_then_speed_is_restored(workspace):
    from recycle_sorter.manipulation.pickplace import pick_at

    arm, gripper = FakeArm(), FakeGripper()
    cfg = {"move_frame": "gripper", "rpc_timeout_s": 5, "arm_speed": 25}
    m = Manipulator(arm, gripper, FakeMotion(), cfg, workspace, {})
    await pick_at(m, 400, 0, TABLE_TOP + 10)
    assert arm.speeds == [8.0, 25.0]


async def test_named_pose_goes_through_the_planner_with_its_full_orientation(workspace):
    motion = FakeMotion()
    m = live_manipulator(workspace, motion, FakeGripper())
    await m.goto_named("survey")
    await m.goto_named("home")  # untaught -> falls back to survey
    assert len(motion.calls) == 2 and motion.calls[0] == motion.calls[1]
    _, frame, x, _, z, o_z, _ = motion.calls[0]
    survey = load_yaml("poses.yaml")["named"]["survey"]  # taught at the table, so not a number to pin here
    assert frame == "world" and (x, z, o_z) == (survey["x"], survey["z"], survey["o_z"])


async def test_service_do_command_matches_original_actions(workspace):
    from recycle_sorter.service import MyGenericService

    assert str(MyGenericService.MODEL) == "hackathons:8ce08475-ece9-436b-acd5-2d213944fb83:generic-service"
    service = MyGenericService("test")
    service.manip = live_manipulator(workspace, FakeMotion(), FakeGripper())
    assert await service.do_command({"action": "static-cycle"}) == {"success": True}
    assert await service.do_command({"action": "go-to-pick"}) == {"success": True}
    assert await service.do_command({"action": "go-to-place"}) == {"success": True}
    assert await service.do_command({"action": "nope"}) == {"error": "unknown command"}
    assert "error" in await service.do_command({"action": "sort"})  # no RobotClient in module mode


async def test_reset_returns_to_the_survey_pose_and_open_gripper_releases(workspace):
    from recycle_sorter.service import MyGenericService

    motion, gripper = FakeMotion(), FakeGripper()
    service = MyGenericService("test")
    service.manip = live_manipulator(workspace, motion, gripper)
    assert await service.do_command({"action": "reset"}) == {"success": True}
    survey = load_yaml("poses.yaml")["named"]["survey"]
    assert (motion.calls[-1][2], motion.calls[-1][4]) == (survey["x"], survey["z"])
    assert await service.do_command({"action": "open-gripper"}) == {"success": True}
    assert gripper.events == ["open"]


async def test_moves_are_sent_in_the_configured_frame_with_the_real_walls(workspace):
    """The machine's `world` was recalibrated: poses go out in the arm's base frame, and the planner gets the walls with them."""
    seen = []

    class Motion(FakeMotion):
        async def move(self, component_name, destination, world_state=None, constraints=None, timeout=None):
            seen.append(world_state)
            return await super().move(component_name, destination, world_state, constraints, timeout)

    motion = Motion()
    cfg = {"move_frame": "gripper", "rpc_timeout_s": 5, "reference_frame": "arm_origin"}
    m = Manipulator(None, FakeGripper(), motion, cfg, workspace, load_yaml("poses.yaml"))
    await m.move_to(300, 0, 200)
    assert motion.calls[0][1] == "arm_origin"
    walls = seen[0].obstacles[0]
    assert walls.reference_frame == "arm_origin"
    assert {g.label for g in walls.geometries} == {"real-wall-front", "real-wall-side"}


# --- survey <-> scan by fixed joint angles (poses.yaml joint_moves) ----------------------------

SURVEY_JOINTS = [-91.964, -31.182, -86.094, -149.805, -94.997, 107.931]
SCAN_JOINTS = [-81.862, -27.512, -31.533, -317.105, 64.803, 252.209]  # stored paired with SURVEY_JOINTS


class JointArm(FakeArm):
    def __init__(self, joints):
        super().__init__()
        self.joints, self.moves = list(joints), []

    async def get_joint_positions(self, timeout=None):
        from types import SimpleNamespace

        return SimpleNamespace(values=list(self.joints))

    async def move_to_joint_positions(self, positions, timeout=None):
        self.joints = list(positions.values)
        self.moves.append(list(positions.values))


def joint_manipulator(workspace, arm, motion):
    cfg = {"move_frame": "gripper", "rpc_timeout_s": 5}
    poses = {**load_yaml("poses.yaml"), "joint_moves": {"poses": {"survey": "cans_view", "scan": "cam_lower_pos"},
                                                        "tolerance_deg": 3}}
    joints = {"cans_view": SURVEY_JOINTS, "cam_lower_pos": SCAN_JOINTS}
    return Manipulator(arm, FakeGripper(), motion, cfg, workspace, poses, joints=joints)


async def test_survey_to_scan_and_back_lands_on_the_exact_taught_numbers_every_trip(workspace):
    arm, motion = JointArm(SURVEY_JOINTS), FakeMotion()
    m = joint_manipulator(workspace, arm, motion)
    for _ in range(3):
        await m.goto_named("scan")
        await m.goto_named("survey")
    assert arm.moves == [SCAN_JOINTS, SURVEY_JOINTS] * 3
    assert motion.calls == []  # the planner was never needed
    assert max(abs(a - b) for a, b in zip(SURVEY_JOINTS, SCAN_JOINTS)) < 170  # no joint spins round


async def test_joint_moves_only_from_the_exact_taught_numbers_otherwise_the_planner(workspace):
    elsewhere = [0.0, -40.0, -60.0, 0.0, 90.0, 0.0]  # e.g. just placed a can: not at survey or scan
    arm, motion = JointArm(elsewhere), FakeMotion()
    m = joint_manipulator(workspace, arm, motion)
    await m.goto_named("survey")
    assert arm.moves == [] and len(motion.calls) == 1
    # At the survey POSE but with the base a whole turn round (the cable wound differently): not the
    # taught numbers, so the move down to scan is planned too.
    arm.joints = [SURVEY_JOINTS[0] + 360.0, *SURVEY_JOINTS[1:]]
    await m.goto_named("scan")
    assert arm.moves == [] and len(motion.calls) == 2
    # Within tolerance of the taught numbers: fixed joints.
    arm.joints = [v + 1.5 for v in SURVEY_JOINTS]
    await m.goto_named("scan")
    assert arm.moves == [SCAN_JOINTS] and len(motion.calls) == 2


async def test_without_taught_joints_every_named_move_is_planned(workspace):
    arm, motion = JointArm(SURVEY_JOINTS), FakeMotion()
    m = live_manipulator(workspace, motion, FakeGripper())
    m.arm = arm
    await m.goto_named("scan")
    assert arm.moves == [] and len(motion.calls) == 1


async def test_jaws_that_are_already_open_are_not_opened_again(workspace):
    """A place ends with the jaws open, so the next pick goes straight to its can (open() blocks ~2 s)."""
    from recycle_sorter.manipulation.pickplace import pick_at, place_at

    gripper = FakeGripper()
    m = live_manipulator(workspace, FakeMotion(), gripper)
    for _ in range(2):
        assert await pick_at(m, 400, 0, TABLE_TOP + 40)
        await place_at(m, 300, 100, TABLE_TOP + 40)
    assert gripper.events == ["open", "grab", "open", "grab", "open"]

    gripper = FakeGripper(holds=False)  # a grab that closed on nothing still leaves the jaws closed
    m = live_manipulator(workspace, FakeMotion(), gripper)
    await pick_at(m, 400, 0, TABLE_TOP + 40)
    await pick_at(m, 300, 0, TABLE_TOP + 40)
    assert gripper.events == ["open", "grab", "open", "grab"]
