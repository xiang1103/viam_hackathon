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
from .manipulation.pickplace import pick, place
from .manipulation.safety import UnsafeTarget
from .perception.segment import draw, segment
from .perception.select import choose_next
from .policy.sort_policy import SortPolicy
from .types import Frame

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


def save_debug(frame: Frame, objects, results, name: str) -> Path:
    out = DATA_DIR / "debug"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    cv2.imwrite(str(path), draw(frame, objects, [r.label for r in results]))
    return path


async def run_sort(robot: LiveRobot, mode: str, max_picks: int = 50) -> Counter:
    """survey -> segment -> choose -> classify -> pick -> place, re-surveying after every pick."""
    sort_cfg = load_yaml("sort.yaml")
    workspace = robot.manip.workspace
    classifier, policy = make_classifier(mode, sort_cfg), SortPolicy(sort_cfg, mode)

    sorted_counts: Counter = Counter()
    attempts: list[np.ndarray] = []  # centroids of failed grasps
    failures = 0

    for _ in range(max_picks):
        await robot.manip.goto_named("survey")
        frame = await robot.snapshot()
        save_frame(frame)
        objects, results = await perceive(frame, classifier, workspace)
        save_debug(frame, objects, results, frame.timestamp)
        if not objects:
            log.info("pile is empty - done")
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
        bin_name = policy.bin_for(result)
        log.info("target %s (%.2f) at (%.0f, %.0f) -> %s", result.label, result.confidence, *target.centroid, bin_name)

        try:
            held = await pick(robot.manip, target)
            if not held:
                raise RuntimeError("grasp came up empty")
            await place(robot.manip, bin_name)
        except (UnsafeTarget, KeyError):
            raise  # configuration problems: never retry blindly
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
        sorted_counts[bin_name] += 1

    await robot.manip.goto_named("home")
    return sorted_counts


async def run_replay(frames_dir: Path, mode: str) -> None:
    """Perception only, over saved frames. No robot needed."""
    sort_cfg, workspace = load_yaml("sort.yaml"), load_yaml("workspace.yaml")
    classifier, policy = make_classifier(mode, sort_cfg), SortPolicy(sort_cfg, mode)
    dirs = sorted(p.parent for p in frames_dir.rglob("meta.json"))
    if not dirs:
        raise SystemExit(f"no saved frames under {frames_dir}")
    for d in dirs:
        frame = load_frame(d)
        objects, results = await perceive(frame, classifier, workspace)
        target = choose_next(objects, workspace)
        overlay = save_debug(frame, objects, results, f"replay-{d.name}")
        print(f"\n{d.name}: {len(objects)} object(s) -> {overlay}")
        for o, r in zip(objects, results):
            mark = "*" if o is target else " "
            print(
                f" {mark} {r.label:8s} conf={r.confidence:.2f} at=({o.centroid[0]:.0f},{o.centroid[1]:.0f}) "
                f"size={o.width:.0f}x{o.length:.0f}x{o.height:.0f} yaw={o.yaw_deg:.0f} -> {policy.bin_for(r)}"
            )
