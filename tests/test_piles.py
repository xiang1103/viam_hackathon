import pytest

from recycle_sorter.app import assess
from recycle_sorter.config import load_yaml
from recycle_sorter.perception.segment import segment
from recycle_sorter.policy.piles import REJECT, PileLayout
from recycle_sorter.policy.sort_policy import SortPolicy
from recycle_sorter.types import Classification

from .synthetic import TABLE_TOP, make_frame


@pytest.fixture
def workspace():
    ws = load_yaml("workspace.yaml")
    ws["table_top"] = TABLE_TOP
    ws["sorted_areas"] = {"left": {"x": [200, 560], "y": [200, 360]}, "right": {"x": [200, 560], "y": [-360, -200]}}
    ws["piles"] = {"slot_gap": 20, "min_pitch": 50, "spare_slots": 2, "pile_gap": 30}
    return ws


def policy(mode_cfg=None):
    return SortPolicy({"modes": {"m": mode_cfg or {}}}, "m")


def all_slots(layout):
    return [s for zones in layout.zones.values() for z in zones for s in z.slots]


# --- which pile an item belongs to -------------------------------------------

def test_every_class_is_its_own_pile_unless_grouped():
    p = policy({"groups": {"warm": ["red", "orange"]}})
    assert p.pile_key(Classification("red", 0.9)) == p.pile_key(Classification("orange", 0.9)) == "warm"
    assert p.pile_key(Classification("blue", 0.9)) == "blue"


def test_low_confidence_goes_to_reject():
    assert policy({"min_confidence": 0.6}).pile_key(Classification("red", 0.4)) == REJECT


# --- assess -> plan ------------------------------------------------------------

def test_assessment_counts_classes_and_sizes(workspace):
    scene = [
        {"x": 300, "y": -90, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "red"},
        {"x": 380, "y": 60, "l": 30, "w": 30, "h": 30, "yaw": 0, "color": "red"},
        {"x": 450, "y": -60, "l": 60, "w": 25, "h": 25, "yaw": 30, "color": "blue"},
    ]
    objects = segment(make_frame(scene), workspace)
    labels = {(300, -90): "red", (380, 60): "red", (450, -60): "blue"}
    results = [
        Classification(min(labels.items(), key=lambda kv: abs(kv[0][0] - o.centroid[0]) + abs(kv[0][1] - o.centroid[1]))[1], 1.0)
        for o in objects
    ]
    demand = assess(objects, results, policy())
    assert demand["red"][0] == 2 and demand["blue"][0] == 1
    assert demand["blue"][1] == pytest.approx(60, abs=3)  # largest blue item sets that pile's spacing


def test_plan_makes_one_pile_per_class_sized_by_count(workspace):
    layout = PileLayout(workspace)
    layout.plan({"red": (5, 30.0), "blue": (1, 30.0)})
    assert set(layout.zones) == {"red", "blue", REJECT}  # reject is always reserved
    assert len(layout.zones["red"][0].slots) >= 5 + 2  # seen + spare for buried items
    assert len(layout.zones["red"][0].slots) > len(layout.zones["blue"][0].slots)


def test_bigger_items_get_wider_spacing(workspace):
    layout = PileLayout(workspace)
    layout.plan({"small": (2, 30.0), "big": (2, 90.0)})
    assert layout.zones["small"][0].pitch == 50  # min_pitch
    assert layout.zones["big"][0].pitch == 110  # 90 + slot_gap


def test_piles_never_overlap_and_stay_inside_the_areas(workspace):
    layout = PileLayout(workspace)
    layout.plan({k: (4, 40.0) for k in "abcde"})
    rects = [z.rect for zones in layout.zones.values() for z in zones]
    for i, (ax0, ax1, ay0, ay1) in enumerate(rects):
        for bx0, bx1, by0, by1 in rects[i + 1 :]:
            assert not (ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0)
    for x, y in all_slots(layout):
        assert any(a["x"][0] <= x <= a["x"][1] and a["y"][0] <= y <= a["y"][1] for a in workspace["sorted_areas"].values())
    assert len(set(all_slots(layout))) == len(all_slots(layout))


# --- staying dynamic during the run ---------------------------------------------

def test_class_first_seen_mid_run_gets_a_new_pile(workspace):
    layout = PileLayout(workspace)
    layout.plan({"red": (2, 30.0)})
    key, slot = layout.next_slot("green", 30.0)  # was buried during the assessment
    assert key == "green" and slot in layout.zones["green"][0].slots


def test_full_pile_grows_an_extension(workspace):
    layout = PileLayout(workspace)
    layout.plan({"red": (1, 30.0)})
    capacity = len(layout.zones["red"][0].slots)
    slots = [layout.next_slot("red", 30.0)[1] for _ in range(capacity + 1)]
    assert len(layout.zones["red"]) == 2 and len(set(slots)) == capacity + 1


def test_new_class_goes_to_reject_when_the_areas_are_full(workspace):
    workspace["sorted_areas"] = {"only": {"x": [200, 420], "y": [200, 260]}}  # room for very little
    layout = PileLayout(workspace)
    layout.plan({"red": (1, 30.0)})
    assert layout.next_slot("green", 30.0)[0] == REJECT


# --- safety ---------------------------------------------------------------------

def test_area_overlapping_unsorted_zone_is_rejected(workspace):
    workspace["sorted_areas"]["bad"] = {"x": [300, 400], "y": [100, 250]}
    with pytest.raises(ValueError, match="picked again"):
        PileLayout(workspace).validate()


def test_area_outside_bounds_is_rejected(workspace):
    workspace["sorted_areas"]["far"] = {"x": [600, 760], "y": [200, 300]}
    with pytest.raises(ValueError, match="outside the workspace"):
        PileLayout(workspace).validate()


def test_taught_areas_replace_the_configured_guesses(workspace):
    taught = {"sorted_areas": {"taught": {"x": [250, 450], "y": [220, 330]}}}
    assert list(PileLayout(workspace, taught).areas) == ["taught"]


def test_shipped_config_is_consistent():
    PileLayout(load_yaml("workspace.yaml"), load_yaml("poses.yaml")).validate()
    SortPolicy(load_yaml("sort.yaml"), "color")
