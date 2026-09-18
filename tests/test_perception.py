import numpy as np
import pytest

from recycle_sorter.classify.color_hsv import HSVColorClassifier
from recycle_sorter.config import load_yaml
from recycle_sorter.perception.frames import cam_to_world_matrix
from recycle_sorter.perception.segment import level, segment, table_report
from recycle_sorter.perception.select import choose_next

from .synthetic import CAM_TO_WORLD, SCENE, TABLE_TOP, make_frame


@pytest.fixture
def workspace():
    ws = load_yaml("workspace.yaml")
    ws["table_top"] = TABLE_TOP
    ws["unsorted_zone"] = {"x": [250, 550], "y": [-150, 150]}
    return ws


def nearest(objects, x, y):
    return min(objects, key=lambda o: np.hypot(o.centroid[0] - x, o.centroid[1] - y))


def test_bare_table_report(workspace):
    report = table_report(make_frame([]), workspace)
    assert "z = 0.0 mm" in report and "OK" in report and "NOT OK" not in report


def miscalibrated(frame, tilt_deg=0.0, dz=0.0):
    """The same scene, but with the camera's pose in the frame system wrong by a tilt and a z offset."""
    c, s_ = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
    err = np.eye(4)
    err[:3, :3] = [[c, 0, s_], [0, 1, 0], [-s_, 0, c]]  # pitch about the camera
    t = frame.cam_to_world[:3, 3].copy()
    bad = frame.cam_to_world.copy()
    bad[:3, :3] = err[:3, :3] @ bad[:3, :3]
    bad[:3, 3] = t + [0, 0, dz]
    frame.cam_to_world = bad
    return frame


def test_camera_calibration_off_by_a_few_cm_and_degrees_still_finds_everything(workspace):
    # Seen on the real machine: the camera reported the table ~29 mm low. Absolute
    # "above z=0" thresholds found nothing; heights relative to the seen table must.
    objects = segment(miscalibrated(make_frame(SCENE), tilt_deg=2.0, dz=-29.0), workspace)
    assert len(objects) == 4
    for b in SCENE[:4]:
        o = nearest(objects, b["x"], b["y"])
        assert o.height == pytest.approx(b["h"], abs=2.5)
        # Grasp heights stay in the ARM's coordinates (table_top), not the camera's wrong ones.
        assert o.top_z == pytest.approx(TABLE_TOP + b["h"], abs=2.5)


def test_unaligned_depth_is_called_out(workspace, caplog):
    # A depth stream that is not aligned to color shows up as a table tilted by several degrees.
    _, _, tilt = level(miscalibrated(make_frame(SCENE), tilt_deg=8.0), workspace)
    assert tilt == pytest.approx(8.0, abs=0.5)
    assert "align_color_depth" in caplog.text
    assert "NOT OK" in table_report(miscalibrated(make_frame([]), tilt_deg=8.0), workspace)


def test_table_colored_depth_noise_is_not_an_object_but_a_tall_white_item_is(workspace):
    frame = make_frame(SCENE[:1])
    # Noise bumps like the real white table produced: 25 mm "tall", but they look like table.
    for u, v in [(200, 150), (420, 330), (150, 360)]:
        frame.depth[v - 15 : v + 15, u - 15 : u + 15] = 400 - 25
    assert len(segment(frame, workspace)) == 1

    white_bottle = [{"x": 350, "y": 60, "l": 40, "w": 40, "h": 90, "yaw": 0, "color": "table"}]
    assert len(segment(make_frame(SCENE[:1] + white_bottle), workspace)) == 2


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




async def test_cam_to_world_matrix_recovers_transform():
    async def transform_point(x, y, z):
        return tuple((CAM_TO_WORLD @ np.array([x, y, z, 1.0]))[:3])

    assert np.allclose(await cam_to_world_matrix(transform_point), CAM_TO_WORLD)


# --- grasp point --------------------------------------------------------------------

def carve(frame, x0, x1, y0, y1, h):
    """Remove part of a block's top (world mm rectangle) by restoring bare table there."""
    from .synthetic import INTR
    z = 400 - h
    us = sorted(int(round(INTR.fx * (x - 400) / z + INTR.cx)) for x in (x0, x1))
    vs = sorted(int(round(INTR.fy * (-y) / z + INTR.cy)) for y in (y0, y1))
    frame.depth[vs[0]:vs[1], us[0]:us[1]] = 400
    frame.color[vs[0]:vs[1], us[0]:us[1]] = (200, 205, 210)
    return frame


def test_plain_block_is_grasped_at_its_middle(workspace):
    o = segment(make_frame([{"x": 400, "y": 0, "l": 60, "w": 30, "h": 30, "yaw": 0, "color": "red"}]), workspace)[0]
    assert np.hypot(*(o.grasp_xy - [400, 0])) < 5
    assert o.grasp_width == pytest.approx(30, abs=3)


def test_arch_is_grasped_across_a_solid_leg_not_through_the_hollow(workspace):
    # A 60x30 block lying along x with a 24 mm wide bite out of the middle of one long side.
    frame = make_frame([{"x": 400, "y": 0, "l": 60, "w": 30, "h": 30, "yaw": 0, "color": "red"}])
    o = segment(carve(frame, 388, 412, -15, 3, 30), workspace)[0]
    assert abs(o.grasp_xy[0] - 400) > 14  # moved off the hollow onto a leg...
    assert o.grasp_width == pytest.approx(30, abs=3)  # ...where the block is solid across its full width
