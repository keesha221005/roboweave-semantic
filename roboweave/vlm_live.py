"""Time one real VLM call on a simulated frame.

    python -m roboweave.vlm_live --model gemma3:4b [--timeout 180]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from roboweave.scenarios import build  # noqa: E402
from roboweave.sensors import Camera  # noqa: E402
from roboweave.vlm import PROMPT, OllamaBackend, encode_jpeg, parse_advice  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma3:4b")
ap.add_argument("--timeout", type=float, default=180.0)
ap.add_argument("--runs", type=int, default=2)
args = ap.parse_args()

world, _ = build("unclear")
world.robot.x, world.robot.y, world.robot.theta = 12.0, 7.0, 0.0     # facing the crate
frame, _ = Camera().render(world)
b64 = encode_jpeg(frame)
backend = OllamaBackend(args.model)
for i in range(args.runs):
    t0 = time.monotonic()
    try:
        raw = backend.ask(b64, PROMPT, args.timeout)
    except Exception as e:
        print(f"run {i + 1}: FAILED after {time.monotonic() - t0:.1f} s: {type(e).__name__}: {e}")
        continue
    dt = time.monotonic() - t0
    print(f"run {i + 1}: {dt:.1f} s  raw={raw[:160]!r}")
    print(f"         parsed -> {parse_advice(raw)}")