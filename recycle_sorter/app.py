from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classify.base import Classifier
from .classify.color_hsv import HSVColorClassifier
from .classify.vision_label import VisionLabelClassifier
from .config import viam_credentials  # noqa: F401  (loads .env)
from .config import DATA_DIR, load_yaml
from .io.recorder import load_frame, load_objects, save_frame
from .io.robot import LiveRobot
from .manipulation.pickplace import grasp_pose, pick, place
from .manipulation.safety import UnsafeTarget
from .perception.segment import draw, segment
from .perception.select import choose_next
from .policy.piles import PileLayout
from .policy.sort_policy import SortPolicy
from .policy.zones import resolve
from .types import Classification, Frame, ObjectObservation
from .viz import draw_layout

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3
MAX_ATTEMPTS_PER_SPOT = 2


def make_classifier(mode: str, sort_cfg: dict[str, Any]) -> Classifier:
    kind = (sort_cfg["modes"].get(mode) or {}).get("classifier")
    if kind == "claude_brand":
        import os

        from .classify.brand_claude import ClaudeBrandClassifier

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit("mode %r reads labels with Claude: put ANTHROPIC_API_KEY=... in .env first." % mode)
        return ClaudeBrandClassifier(sort_cfg["modes"][mode])
    if kind == "local_vlm":
        from .classify.label_vlm import LocalLabelClassifier

        return LocalLabelClassifier(sort_cfg["modes"][mode])
    if mode == "color":
        hsv = HSVColorClassifier(sort_cfg["colors"])
        if load_yaml("machine.yaml").get("perception") == "yolo":
            return hsv  # YOLO only finds the item; its label ("bottle") says nothing about color
        # The class is the Viam vision service that found the object; HSV covers objects
        # that did not come from one (OpenCV fallback, frames recorded without a service).
        return VisionLabelClassifier(fallback=hsv)
    raise ValueError(f"no classifier for mode {mode!r} yet")


async def classify_all(objects: list[ObjectObservation], frame: Frame, classifier: Classifier):
    return [await classifier.classify(o, frame) for o in objects]


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


LOOKS = ("once", "when_needed", "every_pick")


def crowded(target: ObjectObservation, others: list[ObjectObservation], clearance: float) -> bool:
    """Was something close enough to the target that picking it may have moved it?"""
    return any(
        np.linalg.norm(target.centroid - o.centroid) - (target.length + o.length) / 2 < clearance for o in others
    )


def save_scan_debug(scan: Frame, seen, results, name: str, scan_only=(), links=None) -> Path:
    """The eye-level picture with each item's crop box and what it was read as.

    Items are numbered as in the survey picture (#i there is #i here). Cans YOLO found only in
    this picture are orange and numbered scan-N. With `links`, which box in each picture every
    read came from is written next to it as <name>-links.json."""
    img = scan.color.copy()

    def mark(box, text, color):
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    for i, (v, r) in enumerate(zip(seen, results)):
        if v.box is None:
            continue
        if v.readable:
            mark(v.box, f"#{i} {r.label} {r.confidence:.2f}", (0, 200, 0))
        else:
            mark(v.box, f"#{i} hidden {v.hidden:.0%}", (0, 0, 255))
    for r in scan_only:
        link = r.meta["link"]
        mark(link["scan_box"], f"{link['can']} {r.label} {r.confidence:.2f}", (0, 140, 255))
    out = DATA_DIR / "debug"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    cv2.imwrite(str(path), img)
    if links is not None:
        (out / f"{name}-links.json").write_text(json.dumps(links, indent=2))
    return path


async def run_sort(
    robot: LiveRobot,
    mode: str,
    max_picks: int = 50,
    look: str | None = None,
    look_only: bool = False,
    wanted: dict[str, int] | None = None,
) -> Counter:
    """Assess the unsorted zone, create piles to fit what is there, then empty it into them.

    1. assess   survey, find and classify everything visible
    2. plan     one pile per class, sized by count and item size, laid out in the sorted areas
    3. sort     pick topmost -> place in its pile's next slot, until the zone is empty.
                Anything an earlier look missed (it was buried) gets a pile made for it on the spot.

    `look` sets how often a new picture is taken (each one costs a camera download):
      once         a single picture; every pick is planned from it. Fastest. Nothing is
                   re-checked: an item that was hidden, nudged or dropped stays where it is.
      when_needed  plan a whole batch from one picture; look again only after a failed grasp,
                   after picking something that had a close neighbour, and once at the end to
                   confirm the zone is empty.
      every_pick   a new picture after every pick. Slowest, most careful.

    `wanted` turns the run into an ORDER: {class: how many}, e.g. {"coke": 2, "sparkling_water": 1}.
    Only those items are picked, the surest reads first, each class set down in its own pile;
    everything else stays where it is. What could not be found is logged as missing.
    """
    look = look or robot.manip.workspace["pick"].get("look", "when_needed")
    if look not in LOOKS:
        raise ValueError(f"look must be one of {LOOKS}, not {look!r}")
    sort_cfg = load_yaml("sort.yaml")
    workspace = robot.manip.workspace
    layout: PileLayout | None = None  # made on the first look, once the zones are known
    classifier, policy = make_classifier(mode, sort_cfg), SortPolicy(sort_cfg, mode)
    record_objects = robot.cfg.get("perception", "depth") == "viam"  # a service cannot be re-run offline

    sorted_counts: Counter = Counter()
    remaining = Counter(wanted or {})  # order mode: what is still to be fetched
    known: list[tuple[np.ndarray, Classification]] = []  # what each spot held at the last look
    attempts: list[np.ndarray] = []  # centroids of failed grasps
    failures = 0
    planned = False
    queue: list[tuple[ObjectObservation, Classification]] = []  # seen in the last picture, not yet picked

    async def take_a_look() -> list[tuple[ObjectObservation, Classification]]:
        """Go to the survey pose, take one picture, and return what is in the unsorted zone."""
        nonlocal workspace, layout, planned
        await robot.manip.goto_named("survey")
        frame = await robot.snapshot()
        if layout is None:
            # First look: with `auto` zones, find the starting pile wherever it is, draw the
            # unsorted zone around it and lay the sorted areas out beyond it. Fixed for the run.
            async def find(ws: dict[str, Any]) -> list[ObjectObservation]:
                robot.manip.workspace = ws
                return await robot.detect(frame)

            workspace, _ = await resolve(workspace, find)
            robot.manip.workspace = workspace
            layout = PileLayout(workspace, robot.manip.poses)
            layout.validate()
        objects = await robot.detect(frame)
        save_frame(frame, objects=objects if record_objects else None)
        if not objects:
            return []
        if getattr(classifier, "needs_scan", False) and objects:
            # Second look, from eye level: WHERE things are came from above, WHAT they are is on
            # their side. Items hidden behind others come back deferred, to be read on a later look.
            await robot.manip.goto_named("scan")
            scan = await robot.snapshot()
            scan.timestamp = f"{frame.timestamp}-scan"
            save_frame(scan)
            results = await classifier.classify_scan(objects, scan, frame, workspace)
            scan_only = getattr(classifier, "last_scan_only", [])
            links = getattr(classifier, "last_links", None)
            log.info("scan: %s", save_scan_debug(scan, classifier.last_views, results, scan.timestamp, scan_only, links))
            for link in links or []:
                log.info("  can %-7s survey box %-22s scan box %-22s views %-15s -> %s %.2f",
                         link["can"], link["survey_box"], link["scan_box"], "+".join(link["views"]) or "none",
                         link["label"], link["confidence"])
            if scan_only:
                log.info("%d can(s) seen only in the scan picture: read, but no position to pick them from this look",
                         len(scan_only))
        elif hasattr(classifier, "classify_batch"):
            results = await classifier.classify_batch(objects, frame)  # all items in one request
        elif getattr(classifier, "remember", False):
            # A slow reader (seconds per item): an item that has not moved keeps what was read last time.
            results = []
            for o in objects:
                seen_before = next((r for c, r in known if np.linalg.norm(c - o.centroid) < 25.0), None)
                results.append(seen_before or await classifier.classify(o, frame))
            known[:] = [(o.centroid, r) for o, r in zip(objects, results)]
        else:
            results = await classify_all(objects, frame, classifier)
        save_debug(frame, objects, results, frame.timestamp, workspace)
        if not planned:
            readable = [(o, r) for o, r in zip(objects, results) if not r.meta.get("defer")]
            demand = assess([o for o, _ in readable], [r for _, r in readable], policy)
            if wanted is not None:  # piles only for what was ordered, and no bigger than the order
                log.info("on the table: %s", {k: n for k, (n, _) in demand.items()})
                demand = {k: (min(n, wanted[k]), size) for k, (n, size) in demand.items() if k in wanted}
            nothing = "nothing in the unsorted zone" if wanted is None else "none of the order is on the table"
            log.info("%s: %s", "assessment" if wanted is None else "to fetch", {k: n for k, (n, _) in demand.items()} or nothing)
            layout.plan(demand)
            log.info("plan: %s", save_layout(workspace, layout, objects, results, "layout-plan"))
            planned = True
        return list(zip(objects, results))

    looks, fresh = 0, False

    for _ in range(max_picks):
        if wanted is not None and not any(remaining.values()):
            log.info("order complete")
            break
        if queue:
            fresh = False
        else:
            if look == "once" and looks:
                log.info("single-picture run finished")
                break
            fresh = True
            looks += 1
            seen = await take_a_look()
            if not seen:
                log.info("unsorted zone is empty - done")
                break
            for o, r in seen:
                log.info("  %-14s %.2f at (%.0f, %.0f)  %s", r.label, r.confidence, *o.centroid, r.meta.get("text_seen", ""))
            if look_only:
                break
            queue = [(o, r) for o, r in seen if not r.meta.get("defer")]
            if wanted is not None:
                queue = [(o, r) for o, r in queue if remaining[policy.pile_key(r)] > 0]
                if not queue:
                    break  # nothing (more) of what was ordered is on the table
            elif len(queue) < len(seen) and (not queue or look == "once"):
                # Hidden items normally wait for a later look, once what is in front has been
                # sorted. With nothing in front left to sort (or no later look), set them aside.
                log.warning("%d item(s) could not be seen from the side: sending them to reject", len(seen) - len(queue))
                queue = [(o, r if not r.meta.get("defer") else Classification("unknown", 0.0)) for o, r in seen]

        blacklist = [
            a for a in attempts
            if sum(np.linalg.norm(a - b) < 40.0 for b in attempts) >= MAX_ATTEMPTS_PER_SPOT
        ]
        candidates = [o for o, _ in queue]
        if wanted is not None:  # the surest read of any ordered class first, not simply the easiest grasp
            best = max(r.confidence for _, r in queue)
            candidates = [o for o, r in queue if r.confidence >= best - 1e-9]
        target = choose_next(candidates, workspace, blacklist)
        if target is None and wanted is not None:
            target = choose_next([o for o, _ in queue], workspace, blacklist)
        if target is None:
            if fresh or look == "once":
                log.warning("%d object(s) left but none graspable (too wide or given up on)", len(queue))
                break
            queue = []  # the picture is old: look again before giving up
            continue
        result = next(r for o, r in queue if o is target)
        queue = [(o, r) for o, r in queue if o is not target]
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
            if look != "once":
                queue = []  # the item may now be anywhere: look again
            continue

        failures = 0
        sorted_counts[key] += 1
        if wanted is not None:
            remaining[policy.pile_key(result)] -= 1
            queue = [(o, r) for o, r in queue if remaining[policy.pile_key(r)] > 0]
        save_layout(workspace, layout)
        if look == "every_pick":
            queue = []
        elif look == "when_needed" and crowded(target, [o for o, _ in queue], workspace["pick"]["disturb_clearance"]):
            log.info("a neighbour was close to that pick - taking a new picture")
            queue = []

    if wanted is not None and any(remaining.values()):
        log.warning("missing from the order: %s", {k: n for k, n in remaining.items() if n > 0})
    log.info("pictures taken: %d", looks)
    if layout is not None:
        log.info("piles: %s", layout.summary())
    await robot.manip.goto_named("home")
    return sorted_counts


async def run_replay(frames_dir: Path, mode: str) -> None:
    """Assessment and pile plan for each saved frame. No robot needed."""
    sort_cfg, raw_workspace, poses = load_yaml("sort.yaml"), load_yaml("workspace.yaml"), load_yaml("poses.yaml")
    classifier, policy = make_classifier(mode, sort_cfg), SortPolicy(sort_cfg, mode)
    machine = load_yaml("machine.yaml")
    yolo = None
    if machine.get("perception") == "yolo":
        from .perception.yolo import YoloDetector, observations_from_boxes

        yolo = YoloDetector(machine["yolo"])
    dirs = sorted(p.parent for p in frames_dir.rglob("meta.json"))
    if not dirs:
        raise SystemExit(f"no saved frames under {frames_dir}")
    for d in dirs:
        frame = load_frame(d)
        recorded_here = (d / "objects.npz").exists()

        async def find(ws: dict[str, Any], d=d, frame=frame) -> list[ObjectObservation]:
            recorded = load_objects(d, frame, ws)
            if recorded is not None:
                return recorded
            if yolo is not None:
                return observations_from_boxes(yolo.detect(frame.color), frame, ws)
            return segment(frame, ws)

        # Each frame is assessed as if it were the start of a run: zones, then piles.
        workspace, _ = await resolve(raw_workspace, find)
        objects = await find(workspace)
        recorded = objects if recorded_here else None
        if hasattr(classifier, "classify_batch") and not getattr(classifier, "needs_scan", False):
            results = await classifier.classify_batch(objects, frame)
        else:
            results = await classify_all(objects, frame, classifier)
        layout = PileLayout(workspace, poses)
        layout.validate()
        demand = assess(objects, results, policy)
        layout.plan(demand)
        target = choose_next(objects, workspace)

        source = "recorded Viam vision" if recorded is not None else "YOLO + depth" if yolo is not None else "OpenCV depth segmentation"
        print(f"\n{d.name}: {len(objects)} object(s) from {source}")
        z = workspace["unsorted_zone"]
        print(f"  unsorted zone: x {z['x'][0]:.0f}..{z['x'][1]:.0f}, y {z['y'][0]:.0f}..{z['y'][1]:.0f}")
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
