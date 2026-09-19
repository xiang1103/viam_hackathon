"""Load both models into Ollama ahead of time, so pipeline.py doesn't wait for them.

    ollama serve &                 # or the Ollama app
    python -m llm.warmup &         # ~1 min in the background; then leave Ollama running

Requests ask Ollama to keep the models loaded until it stops (keep_alive -1; override with
OLLAMA_KEEP_ALIVE_MODELS, e.g. "30m"), so this is needed once per Ollama start, not per run.
The vision model also reads the reference photos now, which is most of its first-request time.
pipeline.py starts the same warm-up in the background on its own; running this first just
means even the first order is fast.
"""
from __future__ import annotations

import sys
import time

from llm import classify_image, parse_order


def main() -> None:
    start = time.perf_counter()
    parse_order.parse_order("a coke")
    print(f"{parse_order.MODEL}: ready in {time.perf_counter() - start:.1f} s", flush=True)
    refs = len(classify_image.load_references())
    took = classify_image.warm_up()
    print(f"{classify_image.MODEL}: ready with {refs} reference photos in {took:.1f} s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # Ollama not running / model not pulled
        sys.exit(str(e))
