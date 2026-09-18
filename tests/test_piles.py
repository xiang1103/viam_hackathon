import pytest

from recycle_sorter.config import load_yaml
from recycle_sorter.policy.piles import Piles
from recycle_sorter.policy.sort_policy import SortPolicy
from recycle_sorter.types import Classification

PILES = ["pile_1", "pile_2", "pile_3", "reject"]


def policy(mode_cfg, names=PILES):
    return SortPolicy({"reject_pile": "reject", "modes": {"m": mode_cfg}}, "m", names)


def c(label, confidence=0.9):
    return Classification(label, confidence)


def test_each_new_class_claims_the_next_free_pile():
    p = policy({"auto": True})
    assert [p.pile_for(c(x)) for x in ["blue", "red", "blue", "green", "red"]] == [
        "pile_1", "pile_2", "pile_1", "pile_3", "pile_2",
    ]
    assert p.legend == {"blue": "pile_1", "red": "pile_2", "green": "pile_3"}


def test_more_classes_than_piles_overflow_to_reject():
    p = policy({"auto": True})
    for label in ["a", "b", "c"]:
        p.pile_for(c(label))
    assert p.pile_for(c("d")) == "reject"
    assert p.pile_for(c("a")) == "pile_1"  # earlier claims are unaffected


def test_low_confidence_goes_to_reject_without_claiming_a_pile():
    p = policy({"auto": True, "min_confidence": 0.6})
    assert p.pile_for(c("red", 0.4)) == "reject"
    assert p.pile_for(c("blue", 0.9)) == "pile_1"


def test_pinned_labels_share_a_pile_and_auto_skips_it():
    p = policy({"auto": True, "map": {"red": "pile_2", "orange": "pile_2"}})
    assert p.pile_for(c("orange")) == p.pile_for(c("red")) == "pile_2"
    assert [p.pile_for(c(x)) for x in ["blue", "green"]] == ["pile_1", "pile_3"]


def test_auto_off_sends_unmapped_labels_to_reject():
    p = policy({"auto": False, "map": {"red": "pile_1"}})
    assert p.pile_for(c("red")) == "pile_1"
    assert p.pile_for(c("blue")) == "reject"


def test_mapping_to_a_pile_that_does_not_exist_fails_at_startup():
    with pytest.raises(ValueError, match="pile_9"):
        policy({"map": {"red": "pile_9"}})


@pytest.fixture
def workspace():
    return load_yaml("workspace.yaml")


def test_items_spread_across_slots_then_wrap(workspace):
    piles = Piles({"piles": {"pile_1": {"x": 300, "y": 300}}}, workspace)
    slots = [piles.next_slot("pile_1") for _ in range(5)]
    assert len(set(slots[:4])) == 4  # 160 / 70 -> 2x2 distinct slots
    assert all(abs(x - 300) == 35 and abs(y - 300) == 35 for x, y in slots[:4])
    assert slots[4] == slots[0]


def test_untaught_pile_is_a_clear_error(workspace):
    with pytest.raises(KeyError, match="01_teach_pose"):
        Piles({}, workspace).next_slot("pile_1")


def test_pile_overlapping_unsorted_zone_is_rejected(workspace):
    piles = Piles({"piles": {"bad": {"x": 400, "y": 100}}}, workspace)
    with pytest.raises(ValueError, match="picked again"):
        piles.validate()


def test_pile_outside_bounds_is_rejected(workspace):
    piles = Piles({"piles": {"far": {"x": 690, "y": 300}}}, workspace)
    with pytest.raises(ValueError, match="outside the workspace"):
        piles.validate()


def test_shipped_config_is_consistent(workspace):
    piles = Piles(load_yaml("poses.yaml"), workspace)
    piles.validate()
    SortPolicy(load_yaml("sort.yaml"), "color", piles.names)
