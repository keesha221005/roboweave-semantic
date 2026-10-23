"""Run every scenario under every policy and print one comparison table.

    python -m roboweave.compare
    python -m roboweave.compare --weights runs\\detect\\roboweave\\weights\\best.pt
    python -m roboweave.compare --vlm scripted      # simulator-oracle VLM, logic test only
    python -m roboweave.compare --vlm ollama --vlm-model gemma3:4b
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
def _opt(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return val
    return default


_weights = _opt("--weights")
_vlm = _opt("--vlm", "none")
_vlm_model = _opt("--vlm-model")
_latency = float(_opt("--vlm-latency", "1.0"))
sys.argv = sys.argv[:1] + ["--headless"]          # roboweave.navigate parses argv at import

from roboweave.navigate import run  # noqa: E402
from roboweave.scenarios import NAMES  # noqa: E402

TIMEOUT = 60.0


def main():
    print(f"{'scenario':9s} {'policy':10s} {'finished':>9s} {'time(s)':>8s} {'collisions':>10s} "
          f"{'closest(m)':>10s} {'stopped(s)':>10s} {'slowed(s)':>9s}")
    for sc in NAMES:
        for mode in ["roboweave", "box", "stop_all"]:
            r = run(sc, mode, weights=_weights, headless=True, timeout=TIMEOUT, quiet=True,
                    vlm=_vlm if mode == "roboweave" else "none", vlm_model=_vlm_model, vlm_latency=_latency)
            stopped = sum(v for k, v in r["state_time"].items() if k in ("STOP", "YIELD", "WAIT"))
            gap = "-" if r["min_gap"] == float("inf") else f"{r['min_gap']:.2f}"
            t = f"{r['time']:.1f}" if r["time"] else f">{TIMEOUT:.0f}"
            print(f"{sc:9s} {mode:10s} {'yes' if r['time'] else 'NO':>9s} {t:>8s} {r['collisions']:>10d} "
                  f"{gap:>10s} {stopped:>10.1f} {r['state_time'].get('slowed', 0.0):>9.1f}")
    print("\nstopped(s) = time spent stopped, yielding or waiting. stop_all never leaves a pillar, so its time is a floor.")


if __name__ == "__main__":
    main()
