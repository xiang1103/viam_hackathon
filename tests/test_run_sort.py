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
        # The simulated camera is exact: the real arm's camera-to-arm correction would only make it miss.
        workspace["gripper"]["xy_offset"] = [0.0, 0.0]
        machine = {"move_frame": "gripper", "rpc_timeout_s": 5, "arm_speed": 25}
        self.manip = Manipulator(self, self, self, machine, workspace, load_yaml("poses.yaml"))

    # motion service
    async def move(self, component_name, destination, world_state=None, constraints=None, timeout=None):
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


async def test_an_order_fetches_only_what_was_asked_for_and_reports_what_is_missing(caplog):
    robot = SimRobot(SCENE)  # 2 red, 1 blue, 1 green on the table
    with caplog.at_level("INFO", logger="recycle_sorter"):
        counts = await app.run_sort(robot, "color", wanted={"red": 1, "blue": 1, "yellow": 2})

    assert dict(counts) == {"red": 1, "blue": 1}
    assert sorted(color for color, _, _ in robot.placed) == ["blue", "red"]
    assert len(robot.scene) == 2 and robot.holding is None  # the other red and the green were left alone
    assert "missing from the order: {'yellow': 2}" in caplog.text


async def test_a_filled_order_stops_without_another_look():
    robot = SimRobot(SCENE)
    counts = await app.run_sort(robot, "color", wanted={"green": 1})
    assert dict(counts) == {"green": 1} and robot.pictures == 1


async def test_scan_first_takes_the_label_picture_before_the_survey(monkeypatch):
    """pick.scan_first: the survey picture - where the grasp comes from - is the last thing before a pick."""
    from recycle_sorter.types import Classification

    class ScanReader:
        needs_scan = True
        last_views: list = []

        async def classify_scan(self, objects, scan, survey, workspace):
            assert scan is not survey
            return [Classification("red", 0.9) for _ in objects]

    monkeypatch.setattr(app, "make_classifier", lambda mode, cfg: ScanReader())
    monkeypatch.setattr(app, "save_scan_debug", lambda *a, **k: "scan.png")
    for scan_first, expected in ((True, ["scan", "survey"]), (False, ["survey", "scan"])):
        robot = SimRobot(SCENE[:1])
        robot.manip.workspace["pick"]["scan_first"] = scan_first
        visited = []
        goto = robot.manip.goto_named

        async def record(name, goto=goto, visited=visited):
            visited.append(name)
            await goto(name)

        robot.manip.goto_named = record
        await app.run_sort(robot, "color", max_picks=1)
        assert visited[:2] == expected


async def test_an_item_that_is_left_is_logged_with_the_reason(caplog):
    caplog.set_level("WARNING")
    far = {"x": 450, "y": -70, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "red"}  # 455 mm from the arm base
    robot = SimRobot([far])
    robot.manip.workspace["pick"]["max_reach"] = 400
    assert dict(await app.run_sort(robot, "color")) == {} and robot.scene == [far]
    assert "out of reach: 455 mm from the arm base, the limit is 400" in caplog.text
    assert "too wide or given up on" not in caplog.text
