"""pipeline.py end to end against the simulated robot: a typed command in, the ordered cans fetched, the
outcome written next to the pictures. No Ollama, no YOLO model, no hardware: the order parser and the
label reader are fakes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import pipeline
import recycle_sorter.app as app
from llm.parse_order import Order
from recycle_sorter.types import Classification

from .test_brand import CANS
from .test_run_sort import SimRobot

LABEL = {"red": "coke", "blue": "water", "green": "general_soda"}  # what is printed on each simulated can


class ReadsTheCan:
    """Stands in for the VLM: answers with what the simulated can at that spot really is."""

    needs_scan = True
    last_views: list = []

    def __init__(self, robot):
        self.robot = robot

    async def classify_scan(self, objects, scan, survey, workspace):
        out = []
        for o in objects:
            can = min(self.robot.scene, key=lambda b: np.hypot(b["x"] - o.centroid[0], b["y"] - o.centroid[1]))
            out.append(Classification(LABEL[can["color"]], 0.9))
        return out


@pytest.fixture
def table(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "save_frame", lambda *a, **k: tmp_path)
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    robot = SimRobot(CANS)
    named = robot.manip.poses["named"]
    robot.manip.poses = {**robot.manip.poses, "named": {**named, "scan": named["survey"]}}
    monkeypatch.setattr(app, "make_classifier", lambda mode, cfg: ReadsTheCan(robot))
    args = SimpleNamespace(mode="label", max_picks=20, dry_run=False, look=None, pictures=tmp_path / "scans")

    async def get_robot():
        return robot

    return robot, args, get_robot


async def test_a_typed_order_fetches_only_what_was_asked_for_and_skips_what_is_not_there(table, monkeypatch, capsys):
    robot, args, get_robot = table
    on_table = [LABEL[c["color"]] for c in CANS]
    assert on_table.count("coke") >= 1 and "energy_drink" not in on_table
    order = Order(items={"coke": 1, "energy_drink": 2}, not_supported=["orange juice"])
    monkeypatch.setattr(pipeline, "parse_order", lambda text: order)

    await pipeline.handle("a coke, two energy drinks and an orange juice", get_robot, args)

    assert [color for color, _, _ in robot.placed] == ["red"]  # the coke, and nothing else was touched
    assert len(robot.scene) == len(CANS) - 1
    out = capsys.readouterr().out
    assert '"coke": 1' in out and 'missing: {"energy_drink": 2}' in out and "orange juice" in out

    folder = args.pictures
    assert {"survey.png", "scan.png", "order.json"} <= {p.name for p in folder.iterdir()}
    saved = json.loads((folder / "order.json").read_text())
    assert saved[-1]["order"] == {"coke": 1, "energy_drink": 2} and saved[-1]["fetched"] == {"coke": 1}
    assert saved[-1]["missing"] == {"energy_drink": 2} and saved[-1]["not_available"] == ["orange juice"]


async def test_an_order_with_nothing_we_stock_moves_nothing(table, monkeypatch, capsys):
    robot, args, get_robot = table
    monkeypatch.setattr(pipeline, "parse_order", lambda text: Order(items={}, not_supported=["orange juice"]))
    await pipeline.handle("an orange juice", get_robot, args)
    assert robot.pictures == 0 and robot.placed == [] and "nothing to fetch" in capsys.readouterr().out
