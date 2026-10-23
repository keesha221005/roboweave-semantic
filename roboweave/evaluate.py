"""Repeated-trial evaluation of every policy in every scenario.

    python -m roboweave.evaluate --weights runs\\detect\\roboweave\\weights\\best.pt
    python -m roboweave.evaluate --seeds 3                       # quicker
    python -m roboweave.evaluate --scenarios crossing mixed

Each trial changes the lidar noise, the detector noise (simulated detector only) and, for walking
people, their start delay, speed and direction. Seed 0 is the nominal, un-jittered world used by
`compare`. Writes eval/eval_results.json (raw runs + summary) and eval/EVAL.md (tables for the PPT).
All numbers come from simulator ground truth; see roboweave/metrics.py for exact definitions.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from roboweave.scenarios import NAMES  # noqa: E402
from roboweave.session import DT, MODES, run_headless  # noqa: E402

LABEL = {"roboweave": "RoboWeave", "box": "Lidar-only: route around", "stop_all": "Lidar-only: stop & wait"}


def summarise(runs):
    n = len(runs)
    done = [r for r in runs if r["time"]]
    gaps = [r["min_gap"] for r in runs if r["min_gap"] is not None]
    m = lambda k: sum(r["metrics"][k] for r in runs)           # noqa: E731
    stopped = [sum(v for k, v in r["state_time"].items() if k in ("STOP", "YIELD", "WAIT")) for r in runs]
    return {
        "n": n, "finished": len(done),
        "time_mean": round(st.mean(r["time"] for r in done), 1) if done else None,
        "time_min": round(min(r["time"] for r in done), 1) if done else None,
        "time_max": round(max(r["time"] for r in done), 1) if done else None,
        "collisions": sum(r["collisions"] for r in runs),
        "near_misses": m("near_misses"),
        "gap_min": min(gaps) if gaps else None,
        "gap_mean": round(st.mean(gaps), 2) if gaps else None,
        "zone_moving_s": round(m("zone_moving") * DT / n, 1),
        "stopped_mean": round(st.mean(stopped), 1),
        "false_yield_mean": round(m("false_yield_s") / n, 1),
        "stuck_mean": round(m("stuck_s") / n, 1),
        "recall": round(100 * m("vis_hit") / m("vis"), 1) if m("vis") else None,
        "false_person_s": round(m("pers_false") * DT / n, 1),
        "logged": f"{m('logged')}/{m('changes')}",
    }


def f(v, nd=None, unit=""):
    if v is None:
        return "-"
    return (f"{v:.{nd}f}" if nd is not None else str(v)) + unit


def write_md(path, summary, meta):
    L = [f"# RoboWeave evaluation\n",
         f"Detector: **{meta['detector']}**. {meta['seeds']} trials per cell (seed 0 = nominal world, others jitter lidar noise "
         f"and the walking person's start delay, speed and direction). Simulated warehouse, simulator ground truth.\n",
         "## Task and safety\n",
         "| Scenario | Policy | Finished | Time s (mean, min-max) | Collisions | Near misses (<0.5 m) | Closest to a person (m) | Moving inside 2 m person zone (s/trial) |",
         "|---|---|---|---|---|---|---|---|"]
    for sc in meta["scenarios"]:
        for mo in MODES:
            s = summary[sc][mo]
            t = "-" if s["time_mean"] is None else f"{s['time_mean']} ({s['time_min']}-{s['time_max']})"
            L.append(f"| {sc} | {LABEL[mo]} | {s['finished']}/{s['n']} | {t} | {s['collisions']} | {s['near_misses']} | "
                     f"{f(s['gap_min'], 2)} | {s['zone_moving_s']} |")
    L += ["", "Moving inside the zone = seconds per trial the robot was moving while a real person was within 2.0 m in front of it (after 0.5 s); 0.0 also happens when the robot never met a person, so read it with the closest-approach column. "
          "A negative closest approach means contact. People in the simulation walk fixed routes and never avoid the robot, so a robot that stops in their way can be walked into. "
          "Time is averaged over trials that finished.\n",
          "## Efficiency and explainability\n",
          "| Scenario | Policy | Stopped s (mean) | False yield s | Stuck s | Decisions logged |", "|---|---|---|---|---|---|"]
    for sc in meta["scenarios"]:
        for mo in MODES:
            s = summary[sc][mo]
            L.append(f"| {sc} | {LABEL[mo]} | {s['stopped_mean']} | {s['false_yield_mean']} | {s['stuck_mean']} | {s['logged']} |")
    L += ["", "False yield = seconds yielding with no real person within 3.6 m. Stuck = seconds in STOP/WAIT. "
          "Decisions logged = state changes that produced a plain-English log line.\n",
          "## Perception (RoboWeave runs)\n", "| Scenario | Person recall | False person tracks (s per trial) |", "|---|---|---|"]
    for sc in meta["scenarios"]:
        s = summary[sc]["roboweave"]
        L.append(f"| {sc} | {f(s['recall'], 1, '%')} | {s['false_person_s']} |")
    L += ["", "Recall = frames where a real person was visible within 8 m and the tracker had a person within 1 m of them. "
          "False person tracks = seconds per trial that the tracker held a person-class object with no real person within 1 m (in `unclear` with real YOLO, this is where it calls the untrained crate a person).\n"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--scenarios", nargs="*", default=list(NAMES))
    ap.add_argument("--vlm", default="none", choices=["none", "scripted", "ollama", "gemini"])
    ap.add_argument("--vlm-model")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default="eval")
    ap.add_argument("--reuse", action="store_true", help="re-aggregate eval/eval_results.json without running anything")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.reuse:
        d = json.load(open(os.path.join(a.out, "eval_results.json")))
        meta, runs = d["meta"], d["runs"]
        summary = {sc: {mo: summarise([r for r in runs if r["scenario"] == sc and r["mode"] == mo]) for mo in MODES}
                   for sc in meta["scenarios"]}
        json.dump({"meta": meta, "summary": summary, "runs": runs}, open(os.path.join(a.out, "eval_results.json"), "w"))
        write_md(os.path.join(a.out, "EVAL.md"), summary, meta)
        print(open(os.path.join(a.out, "EVAL.md"), encoding="utf-8").read())
        return

    runs, total, k, t0 = [], len(a.scenarios) * len(MODES) * a.seeds, 0, time.time()
    for sc in a.scenarios:
        for mo in MODES:
            for seed in range(a.seeds):
                r = run_headless(sc, mo, a.weights, a.vlm if mo == "roboweave" else "none", a.vlm_model,
                                 a.timeout, seed=None if seed == 0 else seed)
                r["seed"] = seed
                runs.append(r)
                k += 1
                print(f"[{k}/{total}] {sc:9s} {mo:10s} seed {seed}: "
                      + (f"{r['time']:.1f}s" if r["time"] else "DNF") + f"  collisions {r['collisions']}  "
                      f"elapsed {time.time() - t0:.0f}s", flush=True)
    summary = {sc: {mo: summarise([r for r in runs if r["scenario"] == sc and r["mode"] == mo]) for mo in MODES}
               for sc in a.scenarios}
    meta = {"detector": "fine-tuned YOLOv8n" if a.weights else "simulated detector (no weights)",
            "weights": a.weights, "seeds": a.seeds, "scenarios": a.scenarios, "vlm": a.vlm}
    with open(os.path.join(a.out, "eval_results.json"), "w") as fh:
        json.dump({"meta": meta, "summary": summary, "runs": runs}, fh)
    write_md(os.path.join(a.out, "EVAL.md"), summary, meta)
    print("\n" + open(os.path.join(a.out, "EVAL.md"), encoding="utf-8").read())
    print(f"saved {a.out}/EVAL.md and {a.out}/eval_results.json")


if __name__ == "__main__":
    main()