"""Does YOLOv8 detect our drawn people? Run before building anything on top of it.

    pip install ultralytics
    python -m roboweave.yolo_check            # uses yolov8n.pt (auto-downloads)
    python -m roboweave.yolo_check --model yolov8s.pt --conf 0.25

For each distance, the robot faces a lone person (then a lone pillar) in an empty
room, the camera frame goes through YOLO, and a detection counts as a hit when it
is class "person" and overlaps the ground-truth box (IoU >= 0.3).
Saves annotated frames to yolo_check_out/ so you can look at them.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from roboweave.sensors import Camera  # noqa: E402
from roboweave.world import Person, Pillar, Robot, World  # noqa: E402

DISTANCES = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
OFFSETS = [-1.0, 0.0, 1.0]     # sideways offsets (m) so we test more than dead-centre


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def scene(kind, dist, off):
    robot = Robot(2.0, 7.0, 0.0)
    tx, ty = 2.0 + dist, 7.0 + off
    obj = Person(tx, ty) if kind == "person" else Pillar(tx, ty)
    return World(30.0, 14.0, [], [obj] if kind == "pillar" else [],
                 [obj] if kind == "person" else [], robot)


class DummyDetector:
    """Stand-in so the harness can be tested without ultralytics."""
    def __call__(self, frame):
        return []


class YoloDetector:
    def __init__(self, model, conf):
        from ultralytics import YOLO
        self.model, self.conf = YOLO(model), conf
        self.names = self.model.names

    def __call__(self, frame):
        r = self.model.predict(frame[:, :, ::-1], conf=self.conf, verbose=False)[0]  # RGB -> BGR
        out = []
        for b in r.boxes:
            out.append((self.names[int(b.cls)], float(b.conf), tuple(b.xyxy[0].tolist())))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--out", default="yolo_check_out")
    args = ap.parse_args()

    det = DummyDetector() if args.dummy else YoloDetector(args.model, args.conf)
    cam = Camera()
    os.makedirs(args.out, exist_ok=True)
    import pygame

    for kind in ("person", "pillar"):
        print(f"\n== {kind} ==  (hit = detector says '{kind}' on it)")
        print(f"{'dist(m)':>8} {'visible':>8} {'hits':>6} {'best conf':>10}  other labels on it")
        for d in DISTANCES:
            vis = hits = 0
            best, others = 0.0, {}
            for off in OFFSETS:
                w = scene(kind, d, off)
                frame, gt = cam.render(w)
                g = next((g for g in gt if g["cls"] == kind), None)
                if g is None:
                    continue
                vis += 1
                dets = det(frame)
                for name, conf, box in dets:
                    if iou(g["bbox"], box) >= 0.3:
                        if name == kind:
                            hits += 1
                            best = max(best, conf)
                            break
                        others[name] = others.get(name, 0) + 1
                if off == 0.0 and d in (2.0, 5.0):
                    pygame.image.save(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))),
                                      f"{args.out}/{kind}_{d:.0f}m.png")
            print(f"{d:8.1f} {vis:8d} {hits:6d} {best:10.2f}  {others if others else '-'}")
    print(f"\nframes saved in {args.out}/")


if __name__ == "__main__":
    sys.exit(main())
