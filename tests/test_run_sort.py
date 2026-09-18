"""End-to-end: the whole run_sort loop against a simulated robot. Nothing here touches hardware."""
import numpy as np
import pytest

import recycle_sorter.app as app
from recycle_sorter.config import load_yaml
from recycle_sorter.manipulation.motion import Manipulator
from recycle_sorter.perception.segment import segment

from .synthetic import TABLE_TOP, make_frame


class SimRobot:
    """Blocks vanish from the scene when grabbed and are recorded where they are released."""

    def __init__(self, scene, grab_fails_at=()):
        self.scene, self.placed, self.holding = list(scene), [], None
        self.pictures = 0
        self.xy, self.grab_fails_at = (0.0, 0.0), list(grab_fails_at)
        self.cfg = {"perception": "depth"}
        workspace = load_yaml("workspace.yaml")
        workspace["table_top"] = TABLE_TOP
        machine = {"move_frame": "gripper", "rpc_timeout_s": 5, "arm_speed": 25}
        self.manip = Manipulator(self, self, self, machine, workspace, load_yaml("poses.yaml"))

    # motion service
    async def move(self, component_name, destination, constraints=None, timeout=None):
        self.xy = (destination.pose.x, destination.pose.y)
        return True

    # gripper
    async def open(self, timeout=None):
        if self.holding:
            self.placed.append((self.holding["color"], *self.xy))
            self.holding = None

    async def grab(self, timeout=None):
        near = [b for b in self.scene if np.hypot(b["x"] - self.xy[0], b["y"] - self.xy[1]) < 25]
        if not near or any(np.hypot(near[0]["x"] - fx, near[0]["y"] - fy) < 5 for fx, fy in self.grab_fails_at):
            return False
        self.holding = near[0]
        self.scene.remove(near[0])
        return True

    # arm
    async def do_command(self, cmd):
        return {}

    async def stop(self):
        pass

    # LiveRobot surface used by run_sort
    async def snapshot(self):
        self.pictures += 1
        frame = make_frame(self.scene)
        frame.timestamp = "sim"
        return frame

    async def detect(self, frame):
        return segment(frame, self.manip.workspace)


@pytest.fixture(autouse=True)
def no_disk_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "save_frame", lambda *a, **k: tmp_path)
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)


SCENE = [
    {"x": 330, "y": -60, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "red"},
    {"x": 420, "y": 40, "l": 30, "w": 30, "h": 30, "yaw": 20, "color": "red"},
    {"x": 450, "y": -70, "l": 60, "w": 25, "h": 25, "yaw": 30, "color": "blue"},
    {"x": 340, "y": 60, "l": 30, "w": 30, "h": 50, "yaw": 0, "color": "green"},
]


async def test_a_full_sort_empties_the_zone_into_one_pile_per_color():
    robot = SimRobot(SCENE)
    counts = await app.run_sort(robot, "color")

    assert robot.scene == [] and robot.holding is None
    assert dict(counts) == {"red": 2, "blue": 1, "green": 1}

    zone, areas = robot.manip.workspace["unsorted_zone"], robot.manip.workspace["sorted_areas"]
    by_color: dict[str, list] = {}
    for color, x, y in robot.placed:
        by_color.setdefault(color, []).append((x, y))
        assert not (zone["x"][0] < x < zone["x"][1] and zone["y"][0] < y < zone["y"][1])  # never back into the zone
        assert any(a["x"][0] <= x <= a["x"][1] and a["y"][0] <= y <= a["y"][1] for a in areas.values())
    assert len(set(by_color["red"])) == 2  # side by side, not on top of each other
    # Piles do not mix: every other color is further from a red slot than the reds are from each other.
    red = np.array(by_color["red"])
    for color in ("blue", "green"):
        assert np.hypot(*(np.array(by_color[color][0]) - red[0])) > np.hypot(*(red[1] - red[0]))


async def test_a_grasp_that_keeps_failing_is_given_up_on_and_the_rest_still_gets_sorted():
    stubborn = SCENE[3]
    robot = SimRobot(SCENE, grab_fails_at=[(stubborn["x"], stubborn["y"])])
    counts = await app.run_sort(robot, "color")
    assert dict(counts) == {"red": 2, "blue": 1}
    assert robot.scene == [stubborn]  # left where it was, not dropped somewhere random


async def test_an_empty_table_is_not_an_error():
    assert dict(await app.run_sort(SimRobot([]), "color")) == {}


# --- how many pictures ---------------------------------------------------------------

@pytest.mark.parametrize("look,pictures", [("once", 1), ("when_needed", 2), ("every_pick", 5)])
async def test_pictures_taken_for_four_spread_out_blocks(look, pictures):
    robot = SimRobot(SCENE)
    counts = await app.run_sort(robot, "color", look=look)
    assert sum(counts.values()) == 4 and robot.scene == []  # every mode sorts them all...
    assert robot.pictures == pictures  # ...they differ in what it costs


async def test_when_needed_looks_again_after_picking_next_to_a_close_neighbour():
    close = [
        {"x": 360, "y": 0, "l": 30, "w": 30, "h": 50, "yaw": 0, "color": "green"},  # tallest: picked first
        {"x": 405, "y": 0, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "red"},  # 15 mm from it, edge to edge
    ]
    robot = SimRobot(close)
    await app.run_sort(robot, "color", look="when_needed")
    assert robot.scene == [] and robot.pictures == 3  # start, after the crowded pick, final confirmation


async def test_when_needed_looks_again_after_a_failed_grasp_but_once_never_does():
    stubborn = SCENE[3]
    flexible = SimRobot(SCENE, grab_fails_at=[(stubborn["x"], stubborn["y"])])
    await app.run_sort(flexible, "color", look="when_needed")
    single = SimRobot(SCENE, grab_fails_at=[(stubborn["x"], stubborn["y"])])
    counts = await app.run_sort(single, "color", look="once")
    assert flexible.pictures > 2 and single.pictures == 1
    assert dict(counts) == {"red": 2, "blue": 1} and single.scene == [stubborn]  # skipped it, sorted the rest


async def test_unknown_look_mode_is_refused_before_anything_moves():
    robot = SimRobot(SCENE)
    with pytest.raises(ValueError, match="look must be one of"):
        await app.run_sort(robot, "color", look="sometimes")
    assert robot.pictures == 0 and robot.xy == (0.0, 0.0)
