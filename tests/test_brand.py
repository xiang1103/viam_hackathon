"""Brand mode: the eye-level scan, the Claude request, and a full simulated sort. No network, no hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

import recycle_sorter.app as app
from recycle_sorter.classify.brand_claude import Answer, ClaudeBrandClassifier, Item
from recycle_sorter.config import load_yaml
from recycle_sorter.perception.scan import views
from recycle_sorter.policy.piles import REJECT
from recycle_sorter.policy.sort_policy import SortPolicy
from recycle_sorter.types import Classification, Frame, Intrinsics, ObjectObservation

from .synthetic import TABLE_TOP
from .test_run_sort import SimRobot, no_disk_writes  # noqa: F401  (autouse fixture)

CAN = {"l": 66, "w": 66, "h": 122, "yaw": 0}


def can(x, y):
    """A standing can as the top-down survey would report it."""
    return ObjectObservation(
        mask=np.zeros((1, 1), bool), bbox=(0, 0, 0, 0), crop=np.zeros((0, 0, 3), np.uint8),
        centroid=np.array([x, y], float), top_z=TABLE_TOP + 122, height=122, yaw_deg=0, width=66, length=66,
        area_px=0, hole_ratio=0,
    )


def eye_level_frame():
    """A camera 100 mm above the table at the arm, looking straight along +x."""
    t = np.eye(4)
    t[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # columns: camera x, y, z in world
    t[:3, 3] = [0, 0, TABLE_TOP + 100]
    k = Intrinsics(width=1280, height=720, fx=900, fy=900, cx=640, cy=360)
    return Frame(color=np.full((720, 1280, 3), 255, np.uint8), depth=np.zeros((720, 1280), np.uint16), intrinsics=k, cam_to_world=t)


@pytest.fixture
def workspace():
    ws = load_yaml("workspace.yaml")
    ws["table_top"] = TABLE_TOP
    return ws


# --- which cans can be read from the side ------------------------------------------------

def test_cans_side_by_side_are_all_readable(workspace):
    seen = views([can(400, -80), can(400, 80)], eye_level_frame(), eye_level_frame(), workspace)
    assert all(v.readable and v.hidden == 0 for v in seen)
    left_of_image, right_of_image = seen[1].box[0], seen[0].box[0]
    assert left_of_image < right_of_image  # +y is to the camera's left when it looks along +x


def test_a_can_directly_behind_another_is_hidden_not_misread(workspace):
    near, far = can(350, 0), can(520, 0)
    seen = views([far, near], eye_level_frame(), eye_level_frame(), workspace)
    assert seen[1].readable and seen[1].hidden == 0
    assert not seen[0].readable and seen[0].hidden > 0.9  # its crop would show the near can


def test_a_can_outside_the_picture_is_not_readable(workspace):
    seen = views([can(200, 600)], eye_level_frame(), eye_level_frame(), workspace)
    assert seen[0].box is None and not seen[0].readable


# --- the Claude request ---------------------------------------------------------------------

class FakeClaude:
    """Stands in for anthropic.AsyncAnthropic: records the request, answers from `brand_of(index)`."""

    def __init__(self, brand_of, stop_reason="end_turn"):
        self.brand_of, self.stop_reason, self.requests = brand_of, stop_reason, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self.parse))

    async def parse(self, **kwargs):
        self.requests.append(kwargs)
        indexes = [int(b["text"].split()[1].rstrip(":")) for b in kwargs["messages"][0]["content"] if b["type"] == "text" and b["text"].startswith("Item ")]
        if self.stop_reason == "refusal":
            return SimpleNamespace(stop_reason="refusal", parsed_output=None)
        items = [Item(index=i, brand=self.brand_of(i)[0], confidence=self.brand_of(i)[1], text_seen="") for i in indexes]
        return SimpleNamespace(stop_reason="end_turn", parsed_output=Answer(items=items))


async def test_one_request_carries_every_readable_crop_and_names_are_normalised(workspace):
    claude = FakeClaude(lambda i: [("Coca-Cola", 0.95), ("Dr Pepper", 0.9)][i])
    clf = ClaudeBrandClassifier({"known_brands": ["Coca Cola"]}, client=claude)
    results = await clf.classify_scan([can(400, -80), can(400, 80)], eye_level_frame(), eye_level_frame(), workspace)

    assert [r.label for r in results] == ["coca-cola", "dr-pepper"]
    (req,) = claude.requests  # one call for the whole scan
    assert req["model"] == "claude-opus-5" and req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    assert sum(b["type"] == "image" for b in req["messages"][0]["content"]) == 2
    assert "coca-cola" in req["system"]  # the known name is offered, in its normalised form
    assert clf.seen == ["coca-cola", "dr-pepper"]


async def test_names_from_earlier_looks_are_offered_again_so_piles_do_not_split(workspace):
    claude = FakeClaude(lambda i: ("fanta", 0.9))
    clf = ClaudeBrandClassifier({}, client=claude)
    await clf.classify_scan([can(400, 0)], eye_level_frame(), eye_level_frame(), workspace)
    await clf.classify_scan([can(400, 0)], eye_level_frame(), eye_level_frame(), workspace)
    assert "fanta" not in claude.requests[0]["system"].split("Use one of")[-1] or "Use one of" not in claude.requests[0]["system"]
    assert "Use one of these names when it matches: fanta" in claude.requests[1]["system"]


async def test_hidden_cans_are_deferred_and_never_sent_to_claude(workspace):
    claude = FakeClaude(lambda i: ("sprite", 0.9))
    clf = ClaudeBrandClassifier({}, client=claude)
    results = await clf.classify_scan([can(520, 0), can(350, 0)], eye_level_frame(), eye_level_frame(), workspace)
    assert results[0].meta.get("defer") and results[1].label == "sprite"
    assert sum(b["type"] == "image" for b in claude.requests[0]["messages"][0]["content"]) == 1


async def test_a_refused_request_leaves_items_unknown_rather_than_crashing(workspace):
    clf = ClaudeBrandClassifier({}, client=FakeClaude(lambda i: ("x", 1), stop_reason="refusal"))
    results = await clf.classify_scan([can(400, 0)], eye_level_frame(), eye_level_frame(), workspace)
    assert results[0].label == "unknown" and not results[0].meta.get("defer")


def test_unknown_brand_goes_to_reject_even_when_confident():
    policy = SortPolicy({"modes": {"brand": {}}}, "brand")
    assert policy.pile_key(Classification("unknown", 0.99)) == REJECT
    assert policy.pile_key(Classification("pepsi", 0.99)) == "pepsi"


# --- a full simulated brand sort -------------------------------------------------------------

BRAND = {"red": "coca-cola", "blue": "pepsi", "green": "sprite"}
CANS = [
    {"x": 300, "y": -60, "color": "red", **CAN},
    {"x": 300, "y": 60, "color": "blue", **CAN},
    {"x": 420, "y": -60, "color": "red", **CAN},
    {"x": 420, "y": 60, "color": "green", **CAN},
]


class ByColor:
    """A scan classifier that 'reads' the brand off the simulated scene, hiding chosen cans once."""

    needs_scan = True

    def __init__(self, robot, hide_first_look=()):
        self.robot, self.hide, self.calls, self.last_views = robot, list(hide_first_look), 0, []

    async def classify_scan(self, objects, scan, survey, workspace):
        self.calls += 1
        out = []
        for o in objects:
            block = min(self.robot.scene, key=lambda b: np.hypot(b["x"] - o.centroid[0], b["y"] - o.centroid[1]))
            hidden = self.calls == 1 and (block["x"], block["y"]) in self.hide
            out.append(Classification("unseen", 0.0, {"defer": True}) if hidden else Classification(BRAND[block["color"]], 0.95))
        return out


def brand_robot(monkeypatch, **kwargs):
    robot = SimRobot(CANS)
    robot.manip.poses = {**robot.manip.poses, "named": {**robot.manip.poses["named"], "scan": robot.manip.poses["named"]["survey"]}}
    clf = ByColor(robot, **kwargs)
    monkeypatch.setattr(app, "make_classifier", lambda mode, cfg: clf)
    return robot, clf


async def test_cans_are_sorted_into_one_pile_per_brand(monkeypatch):
    robot, _ = brand_robot(monkeypatch)
    counts = await app.run_sort(robot, "brand")
    assert dict(counts) == {"coca-cola": 2, "pepsi": 1, "sprite": 1} and robot.scene == []
    # Survey + scan at the start; the closing look finds the table empty, so it skips the scan picture.
    assert robot.pictures == 3


async def test_a_can_hidden_on_the_first_look_is_read_and_sorted_on_the_next(monkeypatch):
    robot, clf = brand_robot(monkeypatch, hide_first_look=[(420, 60)])
    counts = await app.run_sort(robot, "brand")
    assert dict(counts) == {"coca-cola": 2, "pepsi": 1, "sprite": 1}  # sprite was the hidden one
    assert clf.calls >= 2 and REJECT not in counts


async def test_with_a_single_look_a_hidden_can_is_set_aside_not_guessed(monkeypatch):
    robot, _ = brand_robot(monkeypatch, hide_first_look=[(420, 60)])
    counts = await app.run_sort(robot, "brand", look="once")
    assert dict(counts) == {"coca-cola": 2, "pepsi": 1, REJECT: 1}


async def test_look_only_reports_and_picks_nothing(monkeypatch):
    robot, _ = brand_robot(monkeypatch)
    counts = await app.run_sort(robot, "brand", look_only=True)
    assert dict(counts) == {} and len(robot.scene) == 4 and robot.placed == []


# --- categories, and reading from the survey picture ----------------------------------------------

CATEGORIES = {"coke": "the red can", "diet coke": "the silver can", "energy-drink": "any energy drink"}


def seen_in_survey(x, y, box):
    o = can(x, y)
    o.bbox = box
    return o


def survey_frame():
    f = eye_level_frame()
    f.color[:] = (30, 30, 200)
    return f


async def test_categories_are_offered_with_their_descriptions_and_names_are_normalised():
    claude = FakeClaude(lambda i: [("diet-coke", 0.9), ("energy-drink", 0.8)][i])
    clf = ClaudeBrandClassifier({"categories": CATEGORIES}, client=claude)
    objects = [seen_in_survey(400, -80, (100, 100, 90, 130)), seen_in_survey(400, 80, (300, 100, 90, 130))]
    results = await clf.classify_batch(objects, survey_frame())
    assert [r.label for r in results] == ["diet-coke", "energy-drink"]
    system = claude.requests[0]["system"]
    assert '"diet-coke": the silver can' in system and '"other"' in system
    assert not clf.needs_scan  # survey view: no second picture


async def test_an_answer_outside_the_categories_lands_in_other_not_a_new_pile():
    claude = FakeClaude(lambda i: ("canada-dry", 0.95))
    clf = ClaudeBrandClassifier({"categories": CATEGORIES}, client=claude)
    results = await clf.classify_batch([seen_in_survey(400, 0, (100, 100, 90, 130))], survey_frame())
    assert results[0].label == "other" and results[0].confidence == 0.95


async def test_small_survey_crops_are_enlarged_before_being_sent():
    import base64

    import cv2

    claude = FakeClaude(lambda i: ("coke", 0.9))
    clf = ClaudeBrandClassifier({"categories": CATEGORIES}, client=claude)
    await clf.classify_batch([seen_in_survey(400, 0, (100, 100, 90, 130))], survey_frame())
    image = next(b for b in claude.requests[0]["messages"][0]["content"] if b["type"] == "image")
    sent = cv2.imdecode(np.frombuffer(base64.b64decode(image["source"]["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert max(sent.shape[:2]) == 640  # a ~130 px can is scaled up so small print is legible


async def test_an_item_with_no_usable_box_is_unknown_rather_than_dropped():
    clf = ClaudeBrandClassifier({"categories": CATEGORIES}, client=FakeClaude(lambda i: ("coke", 0.9)))
    results = await clf.classify_batch([seen_in_survey(400, 0, (0, 0, 0, 0))], survey_frame())
    assert [r.label for r in results] == ["unknown"]
