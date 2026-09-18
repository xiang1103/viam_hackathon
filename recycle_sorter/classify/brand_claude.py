from __future__ import annotations

import base64
import logging
import re
from typing import Any

import anthropic
import cv2
from pydantic import BaseModel, Field

from ..perception.scan import views
from ..types import Classification, Frame, ObjectObservation

log = logging.getLogger(__name__)

UNSEEN = "unseen"  # not visible in the scan picture yet: look again once the items in front are gone
UNKNOWN = "unknown"  # visible, but no brand could be read


class Item(BaseModel):
    index: int = Field(description="The number given before the image.")
    brand: str = Field(description="The pile or brand name, exactly as instructed, or 'unknown'.")
    confidence: float = Field(description="0 to 1. How sure you are of the brand.")
    text_seen: str = Field(description="Any text legible on the item, or an empty string.")


class Answer(BaseModel):
    items: list[Item]


SYSTEM = """You identify the brand of drink containers for a sorting robot. Each image is a crop \
from the robot's camera showing one container (a can or a bottle) standing on a white table; \
neighbouring containers may intrude at the edges - judge the one in the middle.

The robot makes one pile per brand, so the same brand must always get exactly the same name, or \
one brand ends up split across two piles.
{naming}
Use everything visible: logo, wordmark, colors, container shape. A label may be turned partly \
away from the camera, so a brand is often recognisable from its colors and a fragment of its \
logo. If you cannot tell, answer "unknown" with low confidence instead of guessing - an unknown \
item is set aside, while a wrong guess ends up in another brand's pile."""


def _name(brand: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", brand.strip().lower()).strip("-") or UNKNOWN


OTHER = "other"  # readable, but none of the configured categories


class ClaudeBrandClassifier:
    """What each item is, read off its label by Claude.

    view: survey  the labels are visible in the survey picture itself (a tilted camera): each
                  item's own detection box is the crop. One picture per look.
          scan    a second, eye-level picture from the taught `scan` pose (see perception/scan.py).

    categories (optional): a fixed set of piles, each with a description of what belongs in it.
    A category can span brands ("energy-drink") or split one ("coke" vs "diet-coke"). Without
    it, every brand seen gets its own pile.
    """

    def __init__(self, cfg: dict[str, Any], client: anthropic.AsyncAnthropic | None = None):
        self.view: str = cfg.get("view", "survey")
        self.categories: dict[str, str] = {_name(k): v for k, v in (cfg.get("categories") or {}).items()}
        self.model: str = cfg.get("model", "claude-opus-5")
        self.effort: str = cfg.get("effort", "low")
        self.known: list[str] = [_name(b) for b in cfg.get("known_brands", [])]
        self.seen: list[str] = []  # names handed out earlier in this run: later looks must reuse them
        self.client = client or anthropic.AsyncAnthropic()  # ANTHROPIC_API_KEY from the environment / .env
        self.last_views: list = []  # kept for the debug picture

    @property
    def needs_scan(self) -> bool:
        return self.view == "scan"

    def _system(self) -> str:
        names = list(dict.fromkeys(self.known + self.seen))
        if self.categories:
            listing = "\n".join(f'- "{name}": {what}' for name, what in self.categories.items())
            naming = (
                "Sort each item into exactly one of these piles, answering with the name in quotes:\n"
                f"{listing}\n"
                f'- "{OTHER}": you can tell what it is, and it belongs in none of the piles above.\n'
            )
        elif names:
            naming = (
                f"Use one of these names when it matches: {', '.join(names)}. For any other brand, "
                "answer with its common name in lowercase with hyphens for spaces (for example "
                '"dr-pepper"). Treat variants of a brand (diet, zero, cherry) as the same brand unless '
                "the list above names the variant.\n"
            )
        else:
            naming = (
                "Answer with the brand's common name in lowercase with hyphens for spaces (for example "
                '"coca-cola", "dr-pepper"). Treat variants of a brand (diet, zero, cherry) as the same brand.\n'
            )
        return SYSTEM.format(naming=naming)

    async def classify_scan(
        self, objects: list[ObjectObservation], scan: Frame, survey: Frame, workspace: dict[str, Any]
    ) -> list[Classification]:
        seen_from_side = self.last_views = views(objects, scan, survey, workspace)
        results = [
            Classification(UNSEEN, 0.0, {"defer": True, "hidden": v.hidden}) for v in seen_from_side
        ]
        readable = [i for i, v in enumerate(seen_from_side) if v.readable]
        for i, c in (await self._ask({i: seen_from_side[i].crop for i in readable})).items():
            results[i] = c
        return results

    async def classify_batch(self, objects: list[ObjectObservation], frame: Frame) -> list[Classification]:
        """view: survey - read each item from its own crop of the survey picture."""
        pictures = {}
        for i, o in enumerate(objects):
            x, y, w, h = o.bbox
            if w < 16 or h < 16:
                continue
            # A little context round the detection box, so a logo at its edge is not cut in half.
            mx, my = int(0.08 * w), int(0.08 * h)
            pictures[i] = frame.color[max(y - my, 0) : y + h + my, max(x - mx, 0) : x + w + mx]
        answers = await self._ask(pictures)
        return [answers.get(i, Classification(UNKNOWN, 0.0, {})) for i in range(len(objects))]

    async def _ask(self, pictures: dict[int, Any]) -> dict[int, Classification]:
        """One request for all the pictures, so the items are named consistently with each other."""
        readable = list(pictures)
        results: dict[int, Classification] = {}
        if not readable:
            return results

        content: list[dict[str, Any]] = []
        for i in readable:
            picture = pictures[i]
            # A can is only ~100 px wide in the survey picture: enlarge it so small print is legible.
            scale = 640 / max(picture.shape[:2])
            if scale > 1:
                picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            ok, jpg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                continue
            content.append({"type": "text", "text": f"Item {i}:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpg.tobytes()).decode()},
            })
        content.append({"type": "text", "text": f"Say what each of these {len(readable)} items is."})

        response = await self.client.beta.messages.parse(
            model=self.model,
            max_tokens=8000,
            system=self._system(),
            messages=[{"role": "user", "content": content}],
            output_format=Answer,
            output_config={"effort": self.effort},  # a lookup, not a reasoning task: keep the robot moving
            # If a safety classifier declines, the API re-runs the request on a fallback model itself.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            log.warning("brand request was not answered (stop_reason=%s); items left as unknown", response.stop_reason)
            for i in readable:
                results[i] = Classification(UNKNOWN, 0.0, {})
            return results

        for i in readable:
            results[i] = Classification(UNKNOWN, 0.0, {})  # anything the answer skips
        for item in response.parsed_output.items:
            if item.index in readable:
                brand = _name(item.brand)
                results[item.index] = Classification(brand, float(item.confidence), {"text_seen": item.text_seen})
                if self.categories and brand not in (*self.categories, OTHER, UNKNOWN):
                    brand = OTHER  # an answer outside the configured piles
                    results[item.index] = Classification(brand, float(item.confidence), {"text_seen": item.text_seen})
                if brand != UNKNOWN and brand not in self.seen:
                    self.seen.append(brand)
        return results
