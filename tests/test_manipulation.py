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
    return ws


def dry_manipulator(workspace, poses):
    cfg = {"move_frame": "gripper", "rpc_timeout_s": 5}
    return Manipulator(None, None, None, cfg, workspace, poses, dry_run=True)


def test_bounds_reject_targets_outside_workspace(workspace):
    check_target(400, 0, 0, workspace)
    for bad in [(100, 0, 0), (720, 0, 0), (400, 480, 0), (400, 0, TABLE_TOP + 2), (400, 0, 900)]:
        with pytest.raises(UnsafeTarget):
            check_target(*bad, workspace)


def test_grasp_never_goes_below_table_clearance(workspace):
    thin = [{"x": 400, "y": 0, "l": 40, "w": 40, "h": 10, "yaw": 0, "color": "red"}]
    obs = segment(make_frame(thin), workspace)[0]
    _, _, z, _ = grasp_pose(obs, workspace)
    assert z >= TABLE_TOP + workspace["bounds"]["z_min_above_table"]


async def test_dry_run_pick_and_place_sequence(workspace, caplog):
    caplog.set_level("INFO")
    obs = segment(make_frame(SCENE[:1]), workspace)[0]
    m = dry_manipulator(workspace, {})
    assert await pick(m, obs)
    await place(m, 300, 300, TABLE_TOP + 10)
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

    async def move(self, component_name, destination, constraints=None, timeout=None):
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


# What the original hard-coded run_static_cycle sent to motion.move, minus its one
# redundant repeat of the place-lift pose: PICK/PLACE at z=75, approach +100, lift +150.
ORIGINAL_STATIC_CYCLE = [
    (300, -150, 175), (300, -150, 75), (300, -150, 225),
    (300, 150, 225), (300, 150, 75), (300, 150, 225),
]


@pytest.mark.parametrize("tcp_offset", [0, 170])
async def test_static_cycle_commands_the_original_poses(workspace, tcp_offset):
    from recycle_sorter.manipulation.pickplace import static_cycle

    workspace["gripper"]["tcp_offset"] = tcp_offset
    workspace["bounds"]["z_max"] = 600
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
    workspace["gripper"]["tcp_offset"] = 170
    motion = FakeMotion()
    await live_manipulator(workspace, motion, FakeGripper()).move_to(400, 0, -100)
    assert motion.calls[0][4] == 70  # fingertips at -100 -> gripper frame at +70


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
