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
    m = dry_manipulator(workspace, {"bins": {"bin_a": {"x": 300, "y": 300}}})
    assert await pick(m, obs)
    await place(m, "bin_a")
    actions = [r.message.split("] ")[1].split(" to ")[0] for r in caplog.records]
    assert actions == [
        "open gripper", "move", "move linear", "grab", "move linear",  # pick
        "move", "move linear", "open gripper", "move linear",  # place
    ]


async def test_place_rejects_bin_outside_workspace(workspace):
    m = dry_manipulator(workspace, {"bins": {"far": {"x": 900, "y": 0}}})
    with pytest.raises(UnsafeTarget):
        await place(m, "far")


async def test_untaught_bin_is_a_clear_error(workspace):
    with pytest.raises(KeyError, match="01_teach_pose"):
        await place(dry_manipulator(workspace, {}), "bin_a")


def test_frame_round_trips_through_recorder(tmp_path, workspace):
    frame = make_frame(SCENE)
    loaded = load_frame(save_frame(frame, tmp_path))
    assert (loaded.depth == frame.depth).all() and (loaded.color == frame.color).all()
    assert len(segment(loaded, workspace)) == len(segment(frame, workspace))
