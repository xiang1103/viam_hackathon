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


async def test_local_label_reader_maps_its_answers_to_piles():
    """The free local reader (llm/classify_image.py) plugs in as a classifier; here it is a fake - no model runs."""
    from types import SimpleNamespace

    import numpy as np

    from recycle_sorter.classify.label_vlm import LocalLabelClassifier
    from recycle_sorter.policy.sort_policy import REJECT, SortPolicy

    answers = iter([
        SimpleNamespace(label="diet_coke", confidence="high", visible_text="Diet Coke", closest_reference="diet_coke"),
        SimpleNamespace(label="sparkling_water", confidence="low", visible_text="", closest_reference="none"),
        SimpleNamespace(label="not_supported", confidence="high", visible_text="", closest_reference="none"),
    ])
    sent = []

    def read(jpeg: bytes):
        sent.append(jpeg)
        return next(answers)

    cfg = {"modes": {"label": {"classifier": "local_vlm", "min_confidence": 0.6, "groups": {}}}}
    classifier, policy = LocalLabelClassifier(cfg["modes"]["label"], read=read), SortPolicy(cfg, "label")
    can = SimpleNamespace(crop=np.full((140, 110, 3), 200, np.uint8), bbox=(0, 0, 110, 140))
    piles = [policy.pile_key(await classifier.classify(can, None)) for _ in range(3)]
    assert piles == ["diet_coke", REJECT, REJECT]  # sure -> its pile; unsure or not a drink -> reject
    assert all(j[:2] == b"\xff\xd8" for j in sent)  # JPEG bytes, as the reader expects


async def test_local_label_reader_sends_the_top_and_the_closer_view_together(workspace):
    """view: scan - each can goes to the reader as two crops (survey + scan picture) in one request.
    The scan crop is the YOLO box nearest the can's projection, even when the projection is off;
    a can hidden there, or with no YOLO box near it, is read from its survey crop alone; a can that
    has not moved since the last look is not read again (~10 s per read)."""
    from recycle_sorter.classify.label_vlm import VIEW_NAMES, LocalLabelClassifier
    from recycle_sorter.perception.scan import views
    from recycle_sorter.perception.yolo import Box

    calls = []

    def read(images, view_names=None):
        calls.append((images, view_names))
        return SimpleNamespace(label="coke", confidence="high", visible_text="Coca-Cola", closest_reference="coke")

    near, behind, beside, lonely = (seen_in_survey(x, y, (100, 100, 90, 130)) for x, y in
                                    [(350, 0), (520, 0), (400, 150), (380, -170)])
    for o in (near, behind, beside, lonely):
        o.crop = np.full((130, 90, 3), 120, np.uint8)
    frame = survey_frame()
    projected = views([near, beside, lonely], frame, frame, workspace)
    # YOLO in the scan picture finds `near` and `beside`, 25 px right of their projection (calibration
    # is off), and nothing near `lonely`.
    yolo = [Box(x + 25, y, x + w + 25, y + h, "can", 0.8) for x, y, w, h in (projected[0].box, projected[1].box)]
    clf = LocalLabelClassifier({"view": "scan"}, read=read, detect=lambda image: yolo)

    results = await clf.classify_scan([behind, near, beside, lonely], frame, frame, workspace)
    assert [r.label for r in results] == ["coke"] * 4 and not any(r.meta.get("defer") for r in results)
    two_views = [c for c in calls if isinstance(c[0], list)]
    one_view = [c for c in calls if not isinstance(c[0], list)]
    assert len(two_views) == 2 and len(one_view) == 2  # near + beside: both views; behind + lonely: survey only
    assert all(len(imgs) == 2 and names == [VIEW_NAMES["survey"], VIEW_NAMES["scan"]] for imgs, names in two_views)
    b = yolo[0]
    assert clf.last_views[1].box == (b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0)  # snapped to YOLO's box

    again = await clf.classify_scan([seen_in_survey(352, 1, (100, 100, 90, 130))], frame, frame, workspace)
    # The same can, unmoved: remembered, not sent again. `beside` is not in this look's survey
    # items but YOLO still boxes it in the scan picture, so that box is read on its own.
    assert again[0].label == "coke" and again[0].meta["remembered"] and len(calls) == 5
    assert calls[-1][1] is None and [r.meta["link"]["scan_yolo_index"] for r in clf.last_scan_only] == [1]


async def test_every_box_in_either_picture_is_read_and_linked(workspace):
    """A can hidden behind another still gets both views when YOLO boxed it in the scan picture,
    without taking the front can's box; a box found only in the scan picture is read on its own,
    kept apart from the pickable items; every read records which boxes it came from."""
    from recycle_sorter.classify.label_vlm import LocalLabelClassifier
    from recycle_sorter.perception.scan import views
    from recycle_sorter.perception.yolo import Box

    calls = []

    def read(images, view_names=None):
        calls.append(images)
        return SimpleNamespace(label="coke", confidence="high", visible_text="", closest_reference="coke")

    behind, near = seen_in_survey(520, 0, (300, 100, 90, 130)), seen_in_survey(350, 0, (100, 100, 90, 130))
    for o in (behind, near):
        o.crop = np.full((130, 90, 3), 120, np.uint8)
    frame = survey_frame()
    projected = views([behind, near], frame, frame, workspace)
    assert projected[0].hidden > workspace["scan"]["max_hidden"]  # the back can is mostly covered
    (fx, fy, fw, fh), (bx, by, bw, bh) = projected[1].box, projected[0].box
    only_in_scan = Box(1100, 50, 1180, 200, "can", 0.7)
    yolo = [only_in_scan, Box(bx, by, bx + bw, by + bh // 2, "can", 0.6), Box(fx, fy, fx + fw, fy + fh, "can", 0.9)]
    clf = LocalLabelClassifier({"view": "scan"}, read=read, detect=lambda image: yolo)

    results = await clf.classify_scan([behind, near], frame, frame, workspace)

    assert len(results) == 2 and len(calls) == 3  # both survey cans + the scan-only can
    assert [r.meta["link"]["scan_yolo_index"] for r in results] == [1, 2]  # each can keeps its own box
    assert [r.meta["link"]["views"] for r in results] == [["survey", "scan"], ["survey", "scan"]]
    assert [r.meta["link"]["survey_box"] for r in results] == [[300, 100, 90, 130], [100, 100, 90, 130]]
    (extra,) = clf.last_scan_only
    assert extra.meta["link"] == {"can": "scan-0", "survey_box": None, "scan_box": [1100, 50, 80, 150],
                                  "scan_yolo_index": 0, "views": ["scan"]}
    assert [link["can"] for link in clf.last_links] == [0, 1, "scan-0"]


def test_label_mode_reads_from_the_scan_pose_unless_cam_pos():
    from recycle_sorter.app import make_classifier

    sort_cfg = load_yaml("sort.yaml")
    assert make_classifier("label", sort_cfg).needs_scan
    assert not make_classifier("label_cam_pos", sort_cfg).needs_scan
    assert "scan" in load_yaml("poses.yaml")["named"]
