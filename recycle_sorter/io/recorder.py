from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import DATA_DIR
from ..perception.segment import finish, observe
from ..types import Frame, Intrinsics, ObjectObservation

FRAMES_DIR = DATA_DIR / "frames"


def save_frame(frame: Frame, root: Path = FRAMES_DIR, objects: list[ObjectObservation] | None = None) -> Path:
    """Persist a snapshot so perception can be developed and regression-tested without the robot.

    Pass the detected `objects` too when they came from a vision service: a service
    cannot be called offline, so its answer is stored with the frame.
    """
    ts = frame.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = root / ts
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "color.png"), frame.color)
    np.save(out / "depth.npy", frame.depth)
    meta = {
        "intrinsics": asdict(frame.intrinsics),
        "cam_to_world": frame.cam_to_world.tolist(),
        "joints": frame.joints,
        "timestamp": ts,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if objects is not None:
        np.savez_compressed(
            out / "objects.npz",
            labels=np.array([o.source_label or "" for o in objects]),
            **{f"points_{i}": o.points for i, o in enumerate(objects)},
        )
    return out


def load_objects(path: Path, frame: Frame, workspace: dict) -> list[ObjectObservation] | None:
    """The vision-service objects recorded with a frame, rebuilt; None if there are none."""
    if not (path / "objects.npz").exists():
        return None
    saved = np.load(path / "objects.npz")
    rebuilt = [
        observe(saved[f"points_{i}"], frame, workspace, source_label=str(label) or None)
        for i, label in enumerate(saved["labels"])
    ]
    return finish([o for o in rebuilt if o is not None], workspace)


def load_frame(path: Path) -> Frame:
    meta = json.loads((path / "meta.json").read_text())
    return Frame(
        color=cv2.imread(str(path / "color.png")),
        depth=np.load(path / "depth.npy"),
        intrinsics=Intrinsics(**meta["intrinsics"]),
        cam_to_world=np.array(meta["cam_to_world"]),
        joints=meta.get("joints"),
        timestamp=meta.get("timestamp", path.name),
    )
