"""Turn a natural-language request into a pick order using a local LLM (Ollama).

    "I want 3 canada dry and 1 red bull"  ->  {"general_soda": 3, "energy_drink": 1}

Requires a running Ollama server (`ollama serve`) with the model pulled
(`ollama pull qwen2.5:1.5b`). No API key: everything runs locally.

Usage:
    python -m llm.parse_order            # interactive: type one request per line
    python -m llm.parse_order "I want 3 canada dry and 1 red bull"   # one-shot
"""

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("ORDER_MODEL", "qwen2.5:1.5b")

# Categories must match what the vision classifier outputs. The description
# tells the model which everyday phrasings and brands belong to each one.
LABELS: dict[str, str] = {
    "coke": "regular Coca-Cola only (also 'coke', 'coca cola', 'classic coke')",
    "diet_coke": "Diet Coke or Coke Zero (also 'diet coke', 'coke zero', 'sugar-free coke')",
    "water": "plain / still bottled water (also 'water', 'bottle of water', 'water bottle'; "
             "e.g. Dasani, Aquafina, Fiji, Evian, Smartwater, Poland Spring)",
    "sparkling_water": "sparkling / seltzer / carbonated water of any flavor "
                       "(e.g. LaCroix, Bubly, Perrier, Topo Chico, 'lime sparkling water')",
    "energy_drink": "energy drinks (e.g. Red Bull, Monster, Celsius, Rockstar, Bang)",
    "coconut_water": "coconut water (e.g. Vita Coco, Zico)",
    "general_soda": "any other soda / pop that is not Coca-Cola or Diet Coke "
                    "(e.g. Sprite, Pepsi, Fanta, Dr Pepper, Canada Dry ginger ale, root beer)",
}
NOT_SUPPORTED = "not_supported"
MAX_QUANTITY = 20

# A 1.5B model tends to force everything into the nearest category, so these
# correct the common misses using the product text. Applied after the model.
UNSUPPORTED_WORDS = ("juice", "lemonade", "milk", "coffee", "tea", "beer", "wine")
SPARKLING_WORDS = ("sparkling", "seltzer", "carbonated", "fizzy", "soda water", "club soda",
                   "la croix", "lacroix", "bubly", "perrier", "topo chico", "pellegrino")
STILL_WATER_WORDS = ("water", "dasani", "aquafina", "fiji", "evian", "smartwater",
                     "poland spring", "deer park", "voss", "essentia")


def _correct_label(requested: str, label: str) -> str:
    """Fix the model's label for juice (not_supported) and water (still vs sparkling)."""
    text = requested.lower()
    for soda in ("root beer", "ginger beer"):  # sodas, despite the name
        text = text.replace(soda, "soda")
    words = text.replace(",", " ").split()
    if any(w in words for w in UNSUPPORTED_WORDS) or "juice" in text:
        return NOT_SUPPORTED
    if "coconut" in text:
        return label
    if any(s in text for s in SPARKLING_WORDS):
        return "sparkling_water"
    if any(s in text for s in STILL_WATER_WORDS):
        return "water"
    return label


@dataclass
class Order:
    items: dict[str, int] = field(default_factory=dict)  # label -> quantity
    not_supported: list[str] = field(default_factory=list)  # requested, but no matching category


def _schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "requested": {"type": "string"},
                        "label": {"type": "string", "enum": labels + [NOT_SUPPORTED]},
                        "quantity": {"type": "integer", "minimum": 1},
                    },
                    "required": ["requested", "label", "quantity"],
                },
            }
        },
        "required": ["items"],
    }


def _system_prompt(labels: dict[str, str]) -> str:
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in labels.items())
    return (
        "You convert a customer's drink request into a list of items.\n"
        f"Categories:\n{catalog}\n\n"
        "Rules:\n"
        "- One entry per distinct product the customer asks for.\n"
        "- 'requested' is the product exactly as the customer said it.\n"
        f"- 'label' must be one of the categories above, or '{NOT_SUPPORTED}' if the product "
        "does not fit any category (e.g. juice, milk, coffee, beer, snacks).\n"
        "- 'quantity' is how many they want. 'a', 'an', 'one' = 1; 'a couple' = 2; "
        "'a few' = 3. If no number is given, use 1.\n"
        "- If the request asks for nothing, return an empty list.\n\n"
        "Examples:\n"
        "- 'coke zero' -> diet_coke (it is sugar-free, not regular coke)\n"
        "- 'water', 'bottle of water', 'water bottle' -> water "
        "(only SPARKLING water is sparkling_water)\n"
        "- 'orange juice', 'apple juice', 'lemonade' -> not_supported "
        "(juice is not coconut_water)\n"
        "- 'ginger ale', 'pepsi' -> general_soda"
    )


def _chat(text: str, labels: dict[str, str]) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt(labels)},
            {"role": "user", "content": text},
        ],
        "format": _schema(list(labels)),
        "stream": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            reply = json.load(resp)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Can't reach Ollama at {OLLAMA_URL} ({e}). Start it with `ollama serve` "
            f"and make sure the model is pulled: `ollama pull {MODEL}`."
        ) from e
    return json.loads(reply["message"]["content"])


def parse_order(text: str, labels: dict[str, str] = LABELS) -> Order:
    """Parse a request into {label: quantity}. Duplicate labels are summed."""
    order = Order()
    for item in _chat(text, labels).get("items", []):
        label, qty = item.get("label"), item.get("quantity", 1)
        if not isinstance(qty, int) or qty < 1:
            continue
        qty = min(qty, MAX_QUANTITY)
        label = _correct_label(item.get("requested", ""), label)
        if label in labels:
            order.items[label] = order.items.get(label, 0) + qty
        else:
            order.not_supported.append(item.get("requested") or str(label))
    return order


def _print_order(text: str) -> None:
    order = parse_order(text)
    print(json.dumps(order.items)
          + (f"  (not available: {order.not_supported})" if order.not_supported else ""))


def main() -> None:
    if len(sys.argv) > 1:
        _print_order(" ".join(sys.argv[1:]))
        return
    print(f"Order parser ({MODEL}). Type a request; empty line, 'quit' or Ctrl-D to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        _print_order(text)


if __name__ == "__main__":
    main()
