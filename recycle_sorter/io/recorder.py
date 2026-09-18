from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import DATA_DIR
from ..types import Frame, Intrinsics

FRAMES_DIR = DATA_DIR / "frames"


def save_frame(frame: Frame, root: Path = FRAMES_DIR) -> Path:
    """Persist a snapshot so perception can be developed and regression-tested without the robot."""
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
    return out


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
