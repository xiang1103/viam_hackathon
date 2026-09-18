"""Classify a cropped image of one can/bottle into a drink category with a local VLM.

    crop.jpg  ->  {"label": "diet_coke", "confidence": "high", "visible_text": "Diet Coke"}

Requires a running Ollama server (`ollama serve`) with the model pulled
(`ollama pull qwen2.5vl:3b`). No API key: everything runs locally.

Usage:
    python -m llm.classify_image crop1.jpg crop2.png ...

From code (e.g. a Viam camera crop's JPEG/PNG bytes):
    from llm.classify_image import classify_image
    result = classify_image(jpeg_bytes)   # or a file path
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from llm.categories import CATEGORIES, LABELS, NOT_SUPPORTED, label_from_text

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:3b")
CONFIDENCE = ["high", "medium", "low"]

# Names the VLM answers with. "coke" and "coconut_water" share a first token, and
# the 3B model under a strict schema often picks the wrong one of the pair, so
# regular Coke gets a distinct name here and is mapped back afterwards.
_TO_VLM = {"coke": "regular_coke"}
_FROM_VLM = {v: k for k, v in _TO_VLM.items()}
VLM_LABELS = [_TO_VLM.get(name, name) for name in LABELS]
_VISUAL_CATALOG = "\n".join(
    f"- {_TO_VLM.get(name, name)}: {c['visual']}" for name, c in CATEGORIES.items())

SYSTEM_PROMPT = f"""You identify the drink in a photo of a single can, bottle or carton.

Categories:
{_VISUAL_CATALOG}
- {NOT_SUPPORTED}: anything else - juice, milk, tea, coffee, beer, alcohol, a cup, \
a non-drink object, or no drink visible.

How to decide:
1. First read any brand name, logo or words printed on the container and write them in \
'visible_text' (empty string if nothing is readable).
2. Use the text first; if no text is readable, use the container's colors, shape and logo.
3. Coca-Cola products are easy to confuse: RED = regular_coke, SILVER or BLACK = diet_coke.
4. For clear water bottles, look for 'sparkling', 'seltzer' or 'carbonated' on the label: \
if present it is sparkling_water, otherwise water.
5. 'confidence' is high when you read the brand clearly, medium when you decide from \
colors/shape only, low when you are guessing."""

SCHEMA = {
    "type": "object",
    "properties": {
        "visible_text": {"type": "string"},
        "label": {"type": "string", "enum": VLM_LABELS + [NOT_SUPPORTED]},
        "confidence": {"type": "string", "enum": CONFIDENCE},
    },
    "required": ["visible_text", "label", "confidence"],
}


@dataclass
class Classification:
    label: str
    confidence: str  # "high" | "medium" | "low"
    visible_text: str  # brand/words the model read, useful for debugging


def classify_image(image: bytes | str | Path) -> Classification:
    """Classify one crop, given as JPEG/PNG bytes or a file path."""
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What drink is this?",
             "images": [base64.b64encode(data).decode()]},
        ],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            reply = json.load(resp)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Can't reach Ollama at {OLLAMA_URL} ({e}). Start it with `ollama serve` "
            f"and make sure the model is pulled: `ollama pull {MODEL}`."
        ) from e
    out = json.loads(reply["message"]["content"])
    text = out.get("visible_text", "")
    label = _FROM_VLM.get(out.get("label"), out.get("label"))
    confidence = out.get("confidence") if out.get("confidence") in CONFIDENCE else "low"
    # Brand text the model read beats the label it picked (e.g. reads 'Coca-Cola'
    # but answers coconut_water). A readable brand also means high confidence.
    text_label = label_from_text(text)
    # Exception: the plain "Coca-Cola" logo is on Diet Coke / Coke Zero too, so it
    # doesn't overrule the model judging diet_coke from the silver/black can.
    if text_label == "coke" and label == "diet_coke":
        text_label = None
    if text_label:
        label, confidence = text_label, "high"
    return Classification(
        label=label if label in LABELS else NOT_SUPPORTED,
        confidence=confidence,
        visible_text=text,
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m llm.classify_image IMAGE [IMAGE ...]")
    for path in sys.argv[1:]:
        start = time.perf_counter()
        r = classify_image(path)
        print(f"{path}: {r.label} ({r.confidence})  text={r.visible_text!r}  "
              f"[{time.perf_counter() - start:.1f}s]")


if __name__ == "__main__":
    main()
