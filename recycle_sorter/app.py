from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classify.base import Classifier
from .classify.color_hsv import HSVColorClassifier
from .config import DATA_DIR, load_yaml
from .io.recorder import load_frame, save_frame
from .io.robot import LiveRobot
from .manipulation.pickplace import grasp_pose, pick, place
from .manipulation.safety import UnsafeTarget
from .perception.segment import draw, segment
from .perception.select import choose_next
from .policy.piles import PileLayout
from .policy.sort_policy import SortPolicy
from .types import Classification, Frame, ObjectObservation
from .viz import draw_layout

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3
MAX_ATTEMPTS_PER_SPOT = 2


def make_classifier(mode: str, sort_cfg: dict[str, Any]) -> Classifier:
    if mode == "color":
        return HSVColorClassifier(sort_cfg["colors"])
    raise ValueError(f"no classifier for mode {mode!r} yet")


async def perceive(frame: Frame, classifier: Classifier, workspace: dict[str, Any]):
    objects = segment(frame, workspace)
    results = [await classifier.classify(o, frame) for o in objects]
    return objects, results


def assess(
    objects: list[ObjectObservation], results: list[Classification], policy: SortPolicy
) -> dict[str, tuple[int, float]]:
    """What is in the unsorted zone: pile key -> (how many, largest item length in mm)."""
    demand: dict[str, tuple[int, float]] = {}
    for o, r in zip(objects, results):
        key = policy.pile_key(r)
        count, size = demand.get(key, (0, 0.0))
        demand[key] = (count + 1, max(size, o.length))
    return demand


def save_debug(frame: Frame, objects, results, name: str, workspace: dict[str, Any] | None = None) -> Path:
    out = DATA_DIR / "debug"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    cv2.imwrite(str(path), draw(frame, objects, [r.label for r in results], workspace))
    return path


def save_layout(workspace, layout: PileLayout, objects=None, results=None, name: str = "layout") -> Path:
    out = DATA_DIR / "debug"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    cv2.imwrite(str(path), draw_layout(workspace, layout, objects, results))
    return path


async def run_sort(robot: LiveRobot, mode: str, max_picks: int = 50) -> Counter:
    """Assess the unsorted zone, create piles to fit what is there, then empty it into them.

    1. assess   survey, find and classify everything visible
    2. plan     one pile per class, sized by count and item size, laid out in the sorted areas
    3. sort     pick topmost -> place in its pile's next slot -> re-survey, until the zone is empty.
                Anything the first look missed (it was buried) gets a pile made for it on the spot.
    """
    sort_cfg = load_yaml("sort.yaml")
    workspace = robot.manip.workspace
    layout = PileLayout(workspace, robot.manip.poses)
    layout.validate()
    classifier, policy = make_classifier(mode, sort_cfg), SortPolicy(sort_cfg, mode)

    sorted_counts: Counter = Counter()
    attempts: list[np.ndarray] = []  # centroids of failed grasps
    failures = 0
    planned = False

    for _ in range(max_picks):
        await robot.manip.goto_named("survey")
        frame = await robot.snapshot()
        save_frame(frame)
        objects, results = await perceive(frame, classifier, workspace)
        save_debug(frame, objects, results, frame.timestamp, workspace)

        if not planned:
            demand = assess(objects, results, policy)
            log.info("assessment: %s", {k: n for k, (n, _) in demand.items()} or "nothing in the unsorted zone")
            layout.plan(demand)
            log.info("plan: %s", save_layout(workspace, layout, objects, results, "layout-plan"))
            planned = True
        if not objects:
            log.info("unsorted zone is empty - done")
            break

        blacklist = [
            a for a in attempts
            if sum(np.linalg.norm(a - b) < 40.0 for b in attempts) >= MAX_ATTEMPTS_PER_SPOT
        ]
        target = choose_next(objects, workspace, blacklist)
        if target is None:
            log.warning("%d object(s) left but none graspable (too wide or blacklisted)", len(objects))
            break
        result = results[objects.index(target)]
        key = policy.pile_key(result)
        log.info("target %s (%.2f) at (%.0f, %.0f) -> %s", result.label, result.confidence, *target.centroid, key)

        try:
            held = await pick(robot.manip, target)
            if not held:
                raise RuntimeError("grasp came up empty")
            # Claim the slot only now, so a failed grasp does not leave a gap in the pile.
            key, slot = layout.next_slot(key, target.length)
            await place(robot.manip, *slot, grasp_pose(target, workspace)[2])
        except UnsafeTarget:
            raise  # a configuration problem: never retry blindly
        except Exception as e:
            log.warning("pick/place failed: %s", e)
            await robot.manip.open()
            attempts.append(target.centroid)
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                log.error("%d consecutive failures - stopping", failures)
                break
            continue

        failures = 0
        sorted_counts[key] += 1
        save_layout(workspace, layout)

    log.info("piles: %s", layout.summary())
    await robot.manip.goto_named("home")
    return sorted_counts


async def run_replay(frames_dir: Path, mode: str) -> None:
    """Assessment and pile plan for each saved frame. No robot needed."""
    sort_cfg, workspace, poses = load_yaml("sort.yaml"), load_yaml("workspace.yaml"), load_yaml("poses.yaml")
    classifier, policy = make_classifier(mode, sort_cfg), SortPolicy(sort_cfg, mode)
    dirs = sorted(p.parent for p in frames_dir.rglob("meta.json"))
    if not dirs:
        raise SystemExit(f"no saved frames under {frames_dir}")
    for d in dirs:
        frame = load_frame(d)
        objects, results = await perceive(frame, classifier, workspace)
        layout = PileLayout(workspace, poses)  # each frame is assessed as if it were the start of a run
        layout.validate()
        demand = assess(objects, results, policy)
        layout.plan(demand)
        target = choose_next(objects, workspace)

        print(f"\n{d.name}: {len(objects)} object(s)")
        print(f"  assessment: { {k: n for k, (n, _) in demand.items()} }")
        for o, r in zip(objects, results):
            mark = "*" if o is target else " "
            print(
                f"  {mark} {r.label:8s} conf={r.confidence:.2f} at=({o.centroid[0]:.0f},{o.centroid[1]:.0f}) "
                f"size={o.width:.0f}x{o.length:.0f}x{o.height:.0f} yaw={o.yaw_deg:.0f} -> {policy.pile_key(r)}"
            )
        for key, zones in layout.zones.items():
            for z in zones:
                print(f"  pile {key:8s} {len(z.slots)} slots @ {z.pitch:.0f} mm in {z.area}: "
                      f"x {z.rect[0]:.0f}-{z.rect[1]:.0f}, y {z.rect[2]:.0f}-{z.rect[3]:.0f}")
        print("  camera view:", save_debug(frame, objects, results, f"replay-{d.name}", workspace))
        print("  table plan: ", save_layout(workspace, layout, objects, results, f"replay-{d.name}-layout"))
