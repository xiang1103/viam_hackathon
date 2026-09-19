"""Turn a natural-language request into a pick order using a local LLM (Ollama).

    "I want 3 canada dry and 1 red bull"  ->  {"ginger_ale": 3, "energy_drink": 1}

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

from llm.categories import CATEGORIES, NOT_SUPPORTED, label_from_text

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("ORDER_MODEL", "qwen2.5:1.5b")
_keep = os.environ.get("OLLAMA_KEEP_ALIVE_MODELS", "-1")  # -1 = keep loaded until Ollama stops
KEEP_ALIVE = int(_keep) if _keep.lstrip("-").isdigit() else _keep  # a number, or a duration like "30m"

MAX_QUANTITY = 20

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


def _system_prompt(labels: dict[str, dict[str, str]]) -> str:
    catalog = "\n".join(f"- {name}: {c['order']}" for name, c in labels.items())
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
        "- 'ginger ale', 'canada dry' -> ginger_ale\n"
        "- 'pepsi', 'sprite', 'root beer' -> non_listed_drinks"
    )


def _chat(text: str, labels: dict[str, dict[str, str]]) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt(labels)},
            {"role": "user", "content": text},
        ],
        "format": _schema(list(labels)),
        "stream": False,
        "keep_alive": KEEP_ALIVE,
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
    except urllib.error.HTTPError as e:  # Ollama answered, but refused the request
        raise RuntimeError(f"Ollama error {e.code}: {e.read().decode(errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Can't reach Ollama at {OLLAMA_URL} ({e}). Start it with `ollama serve` "
            f"and make sure the model is pulled: `ollama pull {MODEL}`."
        ) from e
    return json.loads(reply["message"]["content"])


def parse_order(text: str, labels: dict[str, dict[str, str]] = CATEGORIES) -> Order:
    """Parse a request into {label: quantity}. Duplicate labels are summed."""
    order = Order()
    for item in _chat(text, labels).get("items", []):
        label, qty = item.get("label"), item.get("quantity", 1)
        if not isinstance(qty, int) or qty < 1:
            continue
        qty = min(qty, MAX_QUANTITY)
        # Product words beat the model's pick (it forces juice/water into the
        # nearest category). Shared with the image classifier.
        label = label_from_text(item.get("requested", "")) or label
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
