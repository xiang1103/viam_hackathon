"""YOLO boxes -> 3D objects. The model itself is not loaded here: boxes are given directly."""
import numpy as np
import pytest

from recycle_sorter.config import load_yaml
from recycle_sorter.perception.select import choose_next
from recycle_sorter.perception.yolo import Box, observations_from_boxes

from .synthetic import INTR, TABLE_TOP, make_frame

CANS = [
    {"x": 330, "y": -60, "l": 66, "w": 66, "h": 122, "yaw": 0, "color": "red"},
    {"x": 450, "y": 50, "l": 66, "w": 66, "h": 122, "yaw": 0, "color": "blue"},
]


@pytest.fixture
def workspace():
    ws = load_yaml("workspace.yaml")
    ws["table_top"] = TABLE_TOP
    ws["unsorted_zone"] = {"x": [200, 600], "y": [-150, 150]}
    return ws


def box_around(b, pad=6, label="bottle", conf=0.8):
    """The image box a detector would draw round a block, slightly loose as real ones are."""
    z = 400 - b["h"]
    us = [INTR.fx * (b["x"] + s * b["l"] / 2 - 400) / z + INTR.cx for s in (-1, 1)]
    vs = [INTR.fy * (-(b["y"] + s * b["w"] / 2)) / z + INTR.cy for s in (-1, 1)]
    return Box(int(min(us)) - pad, int(min(vs)) - pad, int(max(us)) + pad, int(max(vs)) + pad, label, conf)


def test_each_box_becomes_a_3d_object_with_metric_size_and_a_grasp(workspace):
    objects = observations_from_boxes([box_around(c) for c in CANS], make_frame(CANS), workspace)
    assert len(objects) == 2
    for o, c in zip(sorted(objects, key=lambda o: o.centroid[0]), CANS):
        assert np.hypot(o.centroid[0] - c["x"], o.centroid[1] - c["y"]) < 3
        assert o.height == pytest.approx(122, abs=2) and o.width == pytest.approx(66, abs=3)
        assert o.grasp_width == pytest.approx(66, abs=4) and o.source_label == "bottle"
        assert o.detection_confidence == 0.8 and o.crop.size > 0


def test_a_box_with_nothing_standing_in_it_is_dropped(workspace):
    bare_table = Box(50, 50, 120, 120, "bottle", 0.9)
    assert observations_from_boxes([bare_table], make_frame(CANS), workspace) == []


def test_every_dropped_box_is_reported_with_why(workspace):
    """The survey picture draws these: a box YOLO found that did not become an item, and the reason."""
    bare_table = Box(50, 50, 120, 120, "bottle", 0.9)
    workspace["unsorted_zone"] = {"x": [200, 400], "y": [-150, 150]}  # the can at x=450 is now outside
    rejected = []
    objects = observations_from_boxes([bare_table] + [box_around(c) for c in CANS], make_frame(CANS), workspace, rejected)
    assert len(objects) == 1 and round(objects[0].centroid[0]) == 330
    assert [(b is bare_table, why) for b, why in rejected] == [
        (True, "nothing above the table"), (False, "outside the unsorted zone")]


def test_a_neighbour_poking_into_the_box_does_not_move_the_object(workspace):
    wide = box_around(CANS[0], pad=6)
    wide.x1 += 40  # reaches toward, and clips the corner region of, whatever is next to it
    o = observations_from_boxes([wide], make_frame(CANS), workspace)[0]
    assert np.hypot(o.centroid[0] - 330, o.centroid[1] + 60) < 3


def test_an_item_beyond_reach_is_never_chosen_and_is_reported(workspace, caplog):
    objects = observations_from_boxes([box_around(c) for c in CANS], make_frame(CANS), workspace)
    workspace["sorted_layout"]["max_reach"] = 400  # the far can (450, 50) is 453 mm out
    chosen = choose_next(objects, workspace)
    assert round(chosen.centroid[0]) == 330
    assert "will be left" in caplog.text and "(450, 50)" in caplog.text
