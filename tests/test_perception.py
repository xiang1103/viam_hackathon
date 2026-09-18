import numpy as np
import pytest

from recycle_sorter.classify.color_hsv import HSVColorClassifier
from recycle_sorter.config import load_yaml
from recycle_sorter.perception.frames import cam_to_world_matrix
from recycle_sorter.perception.segment import segment, table_height
from recycle_sorter.perception.select import choose_next
from recycle_sorter.policy.sort_policy import SortPolicy
from recycle_sorter.types import Classification

from .synthetic import CAM_TO_WORLD, SCENE, TABLE_TOP, make_frame


@pytest.fixture
def workspace():
    ws = load_yaml("workspace.yaml")
    ws["table_top"] = TABLE_TOP
    ws["pile_roi"] = {"x": [250, 550], "y": [-150, 150]}
    return ws


def nearest(objects, x, y):
    return min(objects, key=lambda o: np.hypot(o.centroid[0] - x, o.centroid[1] - y))


def test_table_height_on_bare_table():
    assert table_height(make_frame([])) == pytest.approx(TABLE_TOP, abs=0.5)


def test_segment_finds_pile_objects_only(workspace):
    objects = segment(make_frame(SCENE), workspace)
    assert len(objects) == 4  # the yellow block is outside the ROI
    for b in SCENE[:4]:
        o = nearest(objects, b["x"], b["y"])
        assert np.hypot(o.centroid[0] - b["x"], o.centroid[1] - b["y"]) < 3.0
        assert o.height == pytest.approx(b["h"], abs=1.5)
        assert o.width == pytest.approx(min(b["l"], b["w"]), abs=3.0)
        assert o.length == pytest.approx(max(b["l"], b["w"]), abs=3.0)


def test_yaw_is_short_axis_direction(workspace):
    objects = segment(make_frame(SCENE), workspace)
    blue = nearest(objects, 330, -60)
    # Long side at 30 deg -> gripper closes across the short side at 120 deg == -60 deg.
    assert blue.yaw_deg == pytest.approx(-60.0, abs=3.0)


def test_choose_next_prefers_topmost_and_respects_blacklist(workspace):
    objects = segment(make_frame(SCENE), workspace)
    first = choose_next(objects, workspace)
    assert first.height == pytest.approx(50, abs=1.5)  # the tall green block
    second = choose_next(objects, workspace, blacklist=[first.centroid])
    assert second is not first


def test_choose_next_skips_objects_wider_than_gripper(workspace):
    wide = [{"x": 400, "y": 0, "l": 120, "w": 100, "h": 30, "yaw": 0, "color": "red"}]
    assert choose_next(segment(make_frame(wide), workspace), workspace) is None


async def test_color_labels(workspace):
    frame = make_frame(SCENE)
    objects = segment(frame, workspace)
    clf = HSVColorClassifier(load_yaml("sort.yaml")["colors"])
    for b in SCENE[:4]:
        result = await clf.classify(nearest(objects, b["x"], b["y"]), frame)
        assert result.label == b["color"]


def test_sort_policy_routes_unknown_labels():
    policy = SortPolicy(load_yaml("sort.yaml"), "color")
    assert policy.bin_for(Classification("red", 0.9)) == "bin_a"
    assert policy.bin_for(Classification("grey", 0.9)) == "reject"


async def test_cam_to_world_matrix_recovers_transform():
    async def transform_point(x, y, z):
        return tuple((CAM_TO_WORLD @ np.array([x, y, z, 1.0]))[:3])

    assert np.allclose(await cam_to_world_matrix(transform_point), CAM_TO_WORLD)
