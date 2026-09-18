import numpy as np
import pytest
from viam.proto.common import GeometriesInFrame, Geometry, PointCloudObject

from recycle_sorter.classify.color_hsv import HSVColorClassifier
from recycle_sorter.classify.vision_label import VisionLabelClassifier
from recycle_sorter.config import load_yaml
from recycle_sorter.io.recorder import load_frame, load_objects, save_frame
from recycle_sorter.perception import viam_vision
from recycle_sorter.perception.pcd import parse_pcd
from recycle_sorter.perception.segment import segment

from .synthetic import CAM_TO_WORLD, TABLE_TOP, make_frame

CAMERA = "cam"


def write_pcd(points_m: np.ndarray, binary: bool = True) -> bytes:
    """A PCD the way Viam writes one: meters, float xyz plus a packed int rgb field."""
    n = len(points_m)
    header = (
        "VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F I\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA {'binary' if binary else 'ascii'}\n"
    ).encode()
    if not binary:
        return header + "\n".join(f"{x} {y} {z} 255" for x, y, z in points_m).encode()
    rec = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "i4")])
    rec["x"], rec["y"], rec["z"], rec["rgb"] = points_m[:, 0], points_m[:, 1], points_m[:, 2], 255
    return header + rec.tobytes()


def to_camera_m(world_mm: np.ndarray) -> np.ndarray:
    r, t = CAM_TO_WORLD[:3, :3], CAM_TO_WORLD[:3, 3]
    return ((world_mm - t) @ r) / 1000.0


def pco(world_points_mm: np.ndarray, label: str = "", frame_name: str = CAMERA) -> PointCloudObject:
    return PointCloudObject(
        point_cloud=write_pcd(to_camera_m(world_points_mm)),
        geometries=GeometriesInFrame(reference_frame=frame_name, geometries=[Geometry(label=label)]),
    )


class FakeVision:
    def __init__(self, objects=(), fail=False):
        self.objects, self.fail = list(objects), fail

    async def get_object_point_clouds(self, camera_name, timeout=None):
        assert camera_name == CAMERA
        if self.fail:
            raise RuntimeError("service offline")
        return self.objects


@pytest.fixture
def workspace():
    ws = load_yaml("workspace.yaml")
    ws["table_top"] = TABLE_TOP
    ws["unsorted_zone"] = {"x": [230, 520], "y": [-150, 150]}
    return ws


BLOCKS = {
    "red": {"x": 300, "y": -90, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "red"},
    "blue": {"x": 450, "y": -60, "l": 60, "w": 25, "h": 25, "yaw": 30, "color": "blue"},
    "outside": {"x": 590, "y": 0, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "yellow"},
}


def block_points(frame, workspace, name):
    """World points of one block, plus a ring of table points like a loose segment would include."""
    wide = {**workspace, "unsorted_zone": {"x": [0, 800], "y": [-500, 500]}}
    b = BLOCKS[name]
    o = min(segment(frame, wide), key=lambda o: np.hypot(o.centroid[0] - b["x"], o.centroid[1] - b["y"]))
    table = np.column_stack([np.random.default_rng(0).uniform(-40, 40, (50, 2)) + [b["x"], b["y"]], np.full(50, TABLE_TOP)])
    return np.vstack([o.points, table])


# --- PCD ------------------------------------------------------------------------

@pytest.mark.parametrize("binary", [True, False])
def test_pcd_in_meters_with_extra_fields_becomes_mm(binary):
    pts_m = np.array([[0.1, -0.05, 0.4], [0.12, 0.0, 0.41], [-0.2, 0.3, 0.39]])
    assert np.allclose(parse_pcd(write_pcd(pts_m, binary)), pts_m * 1000, atol=1e-3)


def test_pcd_already_in_mm_is_left_alone_and_empty_is_empty():
    pts_mm = np.array([[100.0, -50.0, 400.0], [120.0, 0.0, 410.0]])
    assert np.allclose(parse_pcd(write_pcd(pts_mm)), pts_mm, atol=1e-3)
    assert parse_pcd(write_pcd(np.zeros((0, 3)))).shape == (0, 3)


# --- detection --------------------------------------------------------------------

async def test_class_is_the_service_that_found_it_and_geometry_comes_from_the_points(workspace):
    frame = make_frame(list(BLOCKS.values()))
    services = {
        "red": FakeVision([pco(block_points(frame, workspace, "red"))]),
        "blue": FakeVision([pco(block_points(frame, workspace, "blue")), pco(block_points(frame, workspace, "outside"))]),
    }
    objects = await viam_vision.detect(services, CAMERA, frame, workspace)
    by_label = {o.source_label: o for o in objects}
    assert set(by_label) == {"red", "blue"}  # the block outside the unsorted zone is dropped

    blue, b = by_label["blue"], BLOCKS["blue"]
    assert np.hypot(blue.centroid[0] - b["x"], blue.centroid[1] - b["y"]) < 3
    assert blue.height == pytest.approx(25, abs=1.5)  # the top of the cloud, not its center
    assert blue.width == pytest.approx(25, abs=3) and blue.length == pytest.approx(60, abs=3)
    assert blue.yaw_deg == pytest.approx(-60, abs=3)
    assert blue.crop.size > 0 and blue.mask[blue.bbox[1] + blue.bbox[3] // 2, blue.bbox[0] + blue.bbox[2] // 2]


async def test_one_item_seen_by_two_services_is_kept_once_and_a_dead_service_is_skipped(workspace):
    frame = make_frame([BLOCKS["red"]])
    pts = block_points(frame, workspace, "red")
    services = {"red": FakeVision([pco(pts)]), "orange": FakeVision([pco(pts[::2])]), "green": FakeVision(fail=True)}
    objects = await viam_vision.detect(services, CAMERA, frame, workspace)
    assert [o.source_label for o in objects] == ["red"]  # the fuller cloud wins


async def test_label_can_come_from_the_detection_itself(workspace):
    frame = make_frame([BLOCKS["red"]])
    services = {"detector": FakeVision([pco(block_points(frame, workspace, "red"), label="soda_can")])}
    objects = await viam_vision.detect(services, CAMERA, frame, workspace, use_detection_label=True)
    assert [o.source_label for o in objects] == ["soda_can"]


async def test_points_already_in_world_frame_are_not_transformed_again(workspace):
    frame = make_frame([BLOCKS["red"]])
    world = block_points(frame, workspace, "red")
    obj = PointCloudObject(
        point_cloud=write_pcd(world / 1000.0),
        geometries=GeometriesInFrame(reference_frame="world", geometries=[Geometry()]),
    )
    objects = await viam_vision.detect({"red": FakeVision([obj])}, CAMERA, frame, workspace)
    assert np.hypot(objects[0].centroid[0] - 300, objects[0].centroid[1] + 90) < 3


# --- classification and replay -------------------------------------------------------

async def test_vision_label_wins_and_hsv_covers_objects_without_one(workspace):
    frame = make_frame([BLOCKS["blue"]])
    clf = VisionLabelClassifier(fallback=HSVColorClassifier(load_yaml("sort.yaml")["colors"]))
    from_service = (await viam_vision.detect({"plastic": FakeVision([pco(block_points(frame, workspace, "blue"))])}, CAMERA, frame, workspace))[0]
    from_opencv = segment(frame, workspace)[0]
    assert (await clf.classify(from_service, frame)).label == "plastic"
    assert (await clf.classify(from_opencv, frame)).label == "blue"


async def test_vision_objects_are_recorded_with_the_frame_and_replay_offline(tmp_path, workspace):
    frame = make_frame([BLOCKS["red"], BLOCKS["blue"]])
    services = {k: FakeVision([pco(block_points(frame, workspace, k))]) for k in ("red", "blue")}
    objects = await viam_vision.detect(services, CAMERA, frame, workspace)
    path = save_frame(frame, tmp_path, objects=objects)

    replayed = load_objects(path, load_frame(path), workspace)
    assert sorted(o.source_label for o in replayed) == ["blue", "red"]
    assert load_objects(save_frame(frame, tmp_path / "plain"), frame, workspace) is None  # no service -> OpenCV path
