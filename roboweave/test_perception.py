"""Score the perception stack against simulator ground truth.

    python -m roboweave.test_perception                 # noisy fake detector (no YOLO needed)
    python -m roboweave.test_perception --weights runs\\detect\\roboweave\\weights\\best.pt

The fake detector deliberately drops boxes, flips person<->pillar labels and adds
false alarms. If the tracker is doing its job, the tracked output is much cleaner
than the raw detections. The --weights run uses the real YOLO model.
"""
from __future__ import annotations

import argparse
import math
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from roboweave.perception import NoisyGTDetector, Perception, YoloDetector  # noqa: E402
from roboweave.sensors import Camera, Lidar  # noqa: E402
from roboweave.world import default_world  # noqa: E402

DT = 1 / 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()

    world, lidar, cam = default_world(), Lidar(seed=3), Camera()
    world.robot.x, world.robot.y, world.robot.theta = 8.0, 7.0, 0.0
    world.people[1].x, world.people[1].y = 12.0, 7.4
    world.people[1].route = [(12.0, 7.4), (22.0, 7.4)]
    fake = None if a.weights else NoisyGTDetector()
    det = YoloDetector(a.weights) if a.weights else fake
    perc = Perception(det, cam, lidar)

    n_frames = int(a.seconds / DT)
    raw_ok = raw_n = 0
    trk_ok = trk_n = 0
    pers_seen = pers_found = 0
    pos_err, switches, last_cls = [], 0, {}
    speeds = {"person": [], "pillar": []}
    speed_err = []

    for i in range(n_frames):
        world.step(DT, 0.0, 0.314)                # spin ~360 deg in 20 s
        t = i * DT
        ranges = lidar.scan(world)
        frame, gt = cam.render(world)
        if fake:
            fake.gt = gt
        rb = world.robot

        # ground-truth world positions of what the camera can see
        gt_pos = []
        for g in gt:
            ang = rb.theta - g["bearing"]
            gt_pos.append((g["cls"], rb.x + g["dist"] * math.cos(ang), rb.y + g["dist"] * math.sin(ang), g))

        if fake:                                  # raw detector accuracy, for comparison
            for d in fake(frame):
                best = min(gt_pos, key=lambda p: abs(p[3]["bbox"][0] - d.bbox[0]) + abs(p[3]["bbox"][1] - d.bbox[1]),
                           default=None)
                if best and abs(best[3]["bbox"][0] - d.bbox[0]) < 8:
                    raw_n += 1
                    raw_ok += d.cls == best[0]

        obs = perc.update(frame, ranges, rb, t)

        for o in obs:
            if o.coasting:
                continue
            best = min(gt_pos, key=lambda p: math.hypot(p[1] - o.x, p[2] - o.y), default=None)
            if best is None or math.hypot(best[1] - o.x, best[2] - o.y) > 0.8:
                continue                           # false alarm, counted below via class accuracy only on matches
            trk_n += 1
            trk_ok += o.cls == best[0]
            pos_err.append(math.hypot(best[1] - o.x, best[2] - o.y))
            if o.id in last_cls and last_cls[o.id] != o.cls:
                switches += 1
            last_cls[o.id] = o.cls
            if i > 60 and o.vel_valid:
                speeds[best[0]].append(o.speed)
                if best[0] == "person":      # true speed of the nearest real person right now
                    tp = min(world.people, key=lambda q: math.hypot(q.x - best[1], q.y - best[2]))
                    speed_err.append(abs(o.speed - (0.0 if tp._wait > 0 else tp.speed)))

        for cls, x, y, g in gt_pos:               # was every clearly-visible person tracked?
            x1, y1, x2, y2 = g["bbox"]
            if cls == "person" and g["dist"] < 9 and (y2 - y1) > 25:
                pers_seen += 1
                pers_found += any(o.cls == "person" and math.hypot(o.x - x, o.y - y) < 0.8 for o in obs)

    print(f"\nframes: {n_frames}   detector: {'YOLO ' + a.weights if a.weights else 'noisy fake (8% drops, 12% label flips, 3% false alarms)'}")
    if fake and raw_n:
        print(f"raw detector class accuracy : {raw_ok / raw_n:6.1%}  ({raw_n} detections)")
    print(f"tracked class accuracy      : {trk_ok / max(trk_n, 1):6.1%}  ({trk_n} tracked obstacles)")
    print(f"class label switches        : {switches}")
    print(f"person recall (visible, <9m): {pers_found / max(pers_seen, 1):6.1%}  ({pers_seen} person-frames)")
    print(f"position error  mean {np.mean(pos_err):.2f} m   p95 {np.percentile(pos_err, 95):.2f} m")
    if speed_err:
        print(f"person speed error vs truth : mean {np.mean(speed_err):.2f} m/s   p95 {np.percentile(speed_err, 95):.2f} m/s")
    for c, v in speeds.items():
        if v:
            print(f"estimated speed {c:7s}: median {np.median(v):.2f} m/s   (pauses included)")


if __name__ == "__main__":
    main()
