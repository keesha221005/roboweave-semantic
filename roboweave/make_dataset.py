"""Render random warehouse scenes and write a YOLO dataset (images + labels).

    python -m roboweave.make_dataset --train 1200 --val 250

Labels come straight from the simulator's ground truth, so no hand labelling.
Classes: 0 = person, 1 = pillar. Appearance (clothing colours), brightness and
noise are randomised so the detector learns "person shape", not one exact sprite.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from roboweave.sensors import Camera  # noqa: E402
from roboweave.world import Person, Pillar, _circle_rect_hit, default_world  # noqa: E402

CLASSES = ["person", "pillar"]
VEST = [(240, 190, 20), (255, 120, 0), (230, 230, 60), (60, 160, 230), (200, 60, 60), (90, 170, 90)]
PANTS = [(40, 40, 70), (30, 30, 30), (70, 60, 50), (90, 90, 110), (20, 50, 90)]
SKIN = [(225, 180, 150), (190, 140, 105), (140, 95, 65), (100, 65, 45), (240, 200, 170)]
HELMET = [(250, 250, 250), (240, 200, 30), (230, 80, 40), (60, 120, 220)]


def random_scene(rng: random.Random):
    w = default_world()
    w.people, w.pillars = [], []

    def free(x, y, r):
        if x - r < 0 or y - r < 0 or x + r > w.width or y + r > w.height:
            return False
        return not any(_circle_rect_hit(x, y, r, rk) for rk in w.racks)

    while True:
        x, y = rng.uniform(0.5, w.width - 0.5), rng.uniform(0.5, w.height - 0.5)
        if free(x, y, 0.45):
            break
    rb = w.robot
    rb.x, rb.y, rb.theta = x, y, rng.uniform(-math.pi, math.pi)

    placed = []
    def place(obj_factory, r):
        for _ in range(25):
            d = rng.uniform(1.3, 9.0)
            ang = rb.theta + rng.uniform(-0.8, 0.8)
            ox, oy = rb.x + d * math.cos(ang), rb.y + d * math.sin(ang)
            if free(ox, oy, r + 0.1) and all(math.hypot(ox - px, oy - py) > 0.8 for px, py in placed):
                placed.append((ox, oy))
                return obj_factory(ox, oy)
        return None

    for _ in range(rng.randint(0, 3)):
        style = {"vest": rng.choice(VEST), "pants": rng.choice(PANTS),
                 "skin": rng.choice(SKIN), "helmet": rng.choice(HELMET)}
        o = place(lambda ox, oy: Person(ox, oy, style=style), 0.25)
        if o:
            w.people.append(o)
    for _ in range(rng.randint(0, 2)):
        o = place(lambda ox, oy: Pillar(ox, oy, r=rng.uniform(0.25, 0.4)), 0.4)
        if o:
            w.pillars.append(o)
    return w


def augment(frame: np.ndarray, rng: random.Random) -> np.ndarray:
    f = frame.astype(np.float32) * rng.uniform(0.7, 1.25)
    f += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 8), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def label_lines(gt, W, H):
    lines = []
    for g in gt:
        x1, y1, x2, y2 = g["bbox"]
        if x2 - x1 < 6 or y2 - y1 < 10:
            continue
        lines.append(f"{CLASSES.index(g['cls'])} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--val", type=int, default=250)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    cam = Camera()
    root = Path(a.out).resolve()
    for split, n in (("train", a.train), ("val", a.val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        rng = random.Random(a.seed + (0 if split == "train" else 999))
        counts = {c: 0 for c in CLASSES}
        for i in range(n):
            w = random_scene(rng)
            frame, gt = cam.render(w)
            frame = augment(frame, rng)
            lines = label_lines(gt, cam.W, cam.H)
            for ln in lines:
                counts[CLASSES[int(ln.split()[0])]] += 1
            Image.fromarray(frame).save(root / "images" / split / f"{i:05d}.jpg", quality=92)
            (root / "labels" / split / f"{i:05d}.txt").write_text("\n".join(lines))
        print(f"{split}: {n} images, boxes {counts}")

    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print("wrote", root / "data.yaml")


if __name__ == "__main__":
    main()
