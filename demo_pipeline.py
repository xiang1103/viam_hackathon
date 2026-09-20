"""pipeline.py for the demo: the labels come from demo_capture.py instead of the VLM.

    python demo_capture.py            # once, before the demo (reads every label, ~10 s a can)
    python demo_pipeline.py           # then type orders as with pipeline.py; THE ARM MOVES
    python demo_pipeline.py --step    # same flags as pipeline.py

Everything runs as in pipeline.py - the order is parsed, the arm goes to `scan` and then `survey`,
YOLO finds the cans, and positions come from the new survey picture - except that each can's label
is looked up in data/demo_labels.json by position (within MATCH_MM of where it was when captured).
A can that isn't in the file (it was moved or added) is still read by the VLM, as usual.
"""
from __future__ import annotations

import asyncio
import json
import sys

import numpy as np

import pipeline
from demo_capture import LABELS_FILE
from llm import parse_order
from recycle_sorter import app
from recycle_sorter.types import Classification

MATCH_MM = 40.0  # cans are ~66 mm wide, so two can centres are never this close (workspace.yaml same_item_mm)

if not LABELS_FILE.exists():
    sys.exit(f"{LABELS_FILE} is missing: run python demo_capture.py first")
saved = json.loads(LABELS_FILE.read_text())
known = [(np.array([c["x"], c["y"]]), Classification(c["label"], c["confidence"], c["meta"]))
         for c in saved["cans"]]
print(f"labels from {LABELS_FILE} ({saved['time']}): "
      + ", ".join(f"{r.label} at ({p[0]:.0f}, {p[1]:.0f})" for p, r in known))

_make_classifier = app.make_classifier


def make_classifier(mode, sort_cfg):
    """run_sort makes a new classifier for every order: seed each one with the saved labels."""
    classifier = _make_classifier(mode, sort_cfg)
    if hasattr(classifier, "_known"):
        classifier._known = list(known)
        classifier.same_spot_mm = MATCH_MM
        classifier.read_scan_only = False  # never picked, and each would cost a VLM read
    return classifier


app.make_classifier = make_classifier
# Warm only the order model: a vision-model warm-up (~55 s) would sit in Ollama's queue ahead of
# the first order. The VLM is only needed for a can that was moved since the capture.
pipeline.warm_all = lambda say: parse_order.warm_up()

if __name__ == "__main__":
    try:
        asyncio.run(pipeline.main())
    except KeyboardInterrupt:
        sys.exit(130)
