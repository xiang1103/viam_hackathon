from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_json(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def save_yaml(name: str, data: dict[str, Any]) -> None:
    (CONFIG_DIR / name).write_text(yaml.safe_dump(data, sort_keys=False))


def viam_credentials() -> tuple[str, str, str]:
    load_dotenv(ROOT / ".env")
    try:
        return (
            os.environ["ROBOT_ADDRESS"],
            os.environ["API_KEY_ID"],
            os.environ["API_KEY"],
        )
    except KeyError as e:
        raise SystemExit(f"Missing {e.args[0]} - copy .env.example to .env and fill it in.") from e
