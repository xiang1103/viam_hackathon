"""Classify a cropped image of one can/bottle into a drink category with a local VLM.

    crop.jpg  ->  {"label": "diet_coke", "confidence": "high", "visible_text": "Diet Coke"}

Requires a running Ollama server (`ollama serve`) with the model pulled
(`ollama pull qwen2.5vl:7b`). No API key: everything runs locally.

Reference photos: every image in reference_pics/ is sent along with each crop as a
labelled example of what that category looks like on our table. The category comes from
the file or folder name: `coke.jpg`, `diet_coke/1.jpg`, or a brand like `canada_dry/x.jpg`
(-> general_soda). Add more photos any time; they are loaded once per process.

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
MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")
CONFIDENCE = ["high", "medium", "low"]
REFERENCE_DIR = Path(os.environ.get(
    "REFERENCE_DIR", Path(__file__).resolve().parent.parent / "reference_pics"))
REFERENCE_MAX_SIDE = 384  # px: enough to show colors and logo, cheap in image tokens
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
# Ollama's default context (4096 tokens) is too small once reference photos are sent.
# Each image costs ~1000 tokens with this model; the text ~1500.
TOKENS_PER_IMAGE, TOKENS_TEXT = 1200, 2048

# Names the VLM answers with. "coke" and "coconut_water" share a first token, and
# the 3B model under a strict schema often picks the wrong one of the pair, so
# regular Coke gets a distinct name here and is mapped back afterwards.
_TO_VLM = {"coke": "regular_coke"}
_FROM_VLM = {v: k for k, v in _TO_VLM.items()}
VLM_LABELS = [_TO_VLM.get(name, name) for name in LABELS]
_VISUAL_CATALOG = "\n".join(
    f"- {_TO_VLM.get(name, name)}: {c['visual']}" for name, c in CATEGORIES.items())

SYSTEM_PROMPT = f"""You identify the drink in a photo of a single can, bottle or carton.
The photo is a small, slightly blurry crop taken from ABOVE at an angle: you mostly see the \
lid and the upper part of the label, and neighbouring cans may poke into the edges.

Categories:
{_VISUAL_CATALOG}
- {NOT_SUPPORTED}: anything else - juice, milk, tea, coffee, beer, alcohol, a cup, \
a non-drink object, or no drink visible.

How to decide:
1. First read any brand name, logo or words printed on the container and write them in \
'visible_text'. Write ONLY letters you can actually see - never guess or complete a brand \
name. Partial words are fine (e.g. 'spind', 'Di'). Empty string if nothing is readable.
2. You can rely on the text, but should use general knowledge to detect the category for each image \
these images should be common United States brand and pictures. 
3. 'confidence' is high when you read the brand clearly, medium when you decide from \
colors/shape only, low when you are guessing."""

REFERENCE_INSTRUCTIONS = """Before the photo to classify, you are shown reference photos of \
the actual drinks on our table, each labelled with its category. They are sharp side views; \
the photo to classify is a blurry view from above, so compare colors, color bands, logo \
shapes and the can's shape rather than small print. In 'closest_reference' name the \
reference that looks most like the photo, or 'none' if none of them matches. The photo \
may also be a drink that has no reference photo: then decide from the categories above."""


@dataclass
class Reference:
    name: str  # e.g. "canada_dry/ginger_ale"
    label: str  # category
    image_b64: str


_references: list[Reference] | None = None


def _category_for(path: Path) -> str | None:
    """Category from the folder or file name: a category name itself, or a known brand."""
    for name in (path.parent.name, path.stem) if path.parent != REFERENCE_DIR else (path.stem,):
        if name in LABELS:
            return name
        label = label_from_text(name.replace("_", " ").replace("-", " "))
        if label in LABELS:
            return label
    return None


def load_references() -> list[Reference]:
    """Reference photos from REFERENCE_DIR, shrunk once and cached for the process."""
    global _references
    if _references is None:
        _references = []
        if REFERENCE_DIR.is_dir():
            import cv2  # only needed to shrink the (often 4K phone) reference photos

            for path in sorted(REFERENCE_DIR.rglob("*")):
                if path.suffix.lower() not in IMAGE_TYPES:
                    continue
                label = _category_for(path)
                image = cv2.imread(str(path))
                if label is None or image is None:
                    print(f"reference photo skipped (no category / unreadable): {path}",
                          file=sys.stderr)
                    continue
                scale = REFERENCE_MAX_SIDE / max(image.shape[:2])
                if scale < 1:
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                name = str(path.relative_to(REFERENCE_DIR).with_suffix(""))
                _references.append(Reference(name, label, base64.b64encode(jpeg.tobytes()).decode()))
    return _references


def _schema(references: list[Reference]) -> dict:
    properties = {"visible_text": {"type": "string"}}
    if references:
        properties["closest_reference"] = {
            "type": "string", "enum": [r.name for r in references] + ["none"]}
    properties["label"] = {"type": "string", "enum": VLM_LABELS + [NOT_SUPPORTED]}
    properties["confidence"] = {"type": "string", "enum": CONFIDENCE}
    return {"type": "object", "properties": properties, "required": list(properties)}


def _messages(image_b64: str, references: list[Reference]) -> list[dict]:
    """System prompt, then one message per reference photo, then the crop. The prefix is
    identical on every call, so Ollama can reuse it and only process the new crop."""
    system = SYSTEM_PROMPT + ("\n\n" + REFERENCE_INSTRUCTIONS if references else "")
    messages = [{"role": "system", "content": system}]
    for r in references:
        messages.append({"role": "user", "images": [r.image_b64],
                         "content": f"Reference '{r.name}': category {_TO_VLM.get(r.label, r.label)}."})
    messages.append({"role": "user", "images": [image_b64],
                     "content": "Now the photo to classify. What drink is this?"})
    return messages


@dataclass
class Classification:
    label: str
    confidence: str  # "high" | "medium" | "low"
    visible_text: str  # brand/words the model read, useful for debugging
    closest_reference: str = ""  # reference photo the model found most similar, or "none"


def classify_image(image: bytes | str | Path) -> Classification:
    """Classify one crop, given as JPEG/PNG bytes or a file path."""
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    references = load_references()
    body = {
        "model": MODEL,
        "messages": _messages(base64.b64encode(data).decode(), references),
        "format": _schema(references),
        "stream": False,
        "options": {"temperature": 0,
                    "num_ctx": TOKENS_TEXT + TOKENS_PER_IMAGE * (len(references) + 1)},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            reply = json.load(resp)
    except urllib.error.HTTPError as e:  # Ollama answered, but refused the request
        raise RuntimeError(f"Ollama error {e.code}: {e.read().decode(errors='replace')}") from e
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
    # but answers coconut_water). Confidence stays the model's own: on blurry crops
    # it sometimes invents the text, so a keyword match alone doesn't mean "high".
    text_label = label_from_text(text)
    # Exception: the plain "Coca-Cola" logo is on Diet Coke / Coke Zero too, so it
    # doesn't overrule the model judging diet_coke from the silver/black can.
    if text_label == "coke" and label == "diet_coke":
        text_label = None
    if text_label:
        label = text_label
    return Classification(
        label=label if label in LABELS else NOT_SUPPORTED,
        confidence=confidence,
        visible_text=text,
        closest_reference=out.get("closest_reference", ""),
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m llm.classify_image IMAGE [IMAGE ...]")
    refs = load_references()
    print(f"{len(refs)} reference photos: " + ", ".join(f"{r.name} -> {r.label}" for r in refs))
    for path in sys.argv[1:]:
        start = time.perf_counter()
        r = classify_image(path)
        print(f"{path}: {r.label} ({r.confidence})  text={r.visible_text!r}  "
              f"ref={r.closest_reference}  [{time.perf_counter() - start:.1f}s]")


if __name__ == "__main__":
    main()
