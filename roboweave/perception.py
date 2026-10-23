"""Perception: camera detections + lidar ranges -> stable, labelled obstacles.

Pipeline (runs every frame, all local, no network):
    camera frame --YOLO--> detections (class, conf, box)
    detections   --Tracker--> tracks with smoothed class (no person/pillar flicker)
    tracks + lidar --locate--> obstacles with world position and velocity

Lidar gives WHERE (range), camera gives WHAT (class). Neither alone is enough.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from roboweave.sensors import CAM_H

CLASS_RADIUS = {"person": 0.25, "pillar": 0.30}   # prior used to get from surface to centre


# ---------------------------------------------------------------- detection
@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple   # x1, y1, x2, y2 in pixels


class YoloDetector:
    """Fine-tuned YOLOv8 (classes: person, pillar)."""

    def __init__(self, weights: str, conf: float = 0.25):
        from ultralytics import YOLO
        self.model, self.conf = YOLO(weights), conf
        self.names = self.model.names

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame[:, :, ::-1], conf=self.conf, verbose=False)[0]  # RGB -> BGR
        return [Detection(self.names[int(b.cls)], float(b.conf), tuple(b.xyxy[0].tolist()))
                for b in r.boxes]


import random  # noqa: E402


class NoisyGTDetector:
    """Development stand-in for YOLO: ground-truth boxes with drops, label flips, false alarms."""

    def __init__(self, seed=1, drop=0.08, flip=0.12, false_pos=0.03):
        self.rng = random.Random(seed)
        self.drop, self.flip, self.false_pos = drop, flip, false_pos
        self.gt = []

    def __call__(self, frame):
        r, out = self.rng, []
        for g in self.gt:
            if r.random() < self.drop:
                continue
            cls, conf = g["cls"], r.uniform(0.85, 0.97)
            if r.random() < self.flip:
                cls, conf = ("pillar" if cls == "person" else "person"), r.uniform(0.5, 0.7)
            x1, y1, x2, y2 = [v + r.uniform(-3, 3) for v in g["bbox"]]
            out.append(Detection(cls, conf, (x1, y1, x2, y2)))
        if r.random() < self.false_pos:
            x, y = r.uniform(0, 560), r.uniform(0, 250)
            out.append(Detection(r.choice(["person", "pillar"]), r.uniform(0.3, 0.6), (x, y, x + 40, y + 90)))
        return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------------ tracker
@dataclass
class Track:
    id: int
    bbox: tuple
    scores: dict                 # smoothed per-class evidence
    cls: str
    hits: int = 1
    misses: int = 0
    last_seen: float = 0.0
    pos: tuple | None = None     # smoothed world (x, y)
    hist: deque = field(default_factory=lambda: deque(maxlen=15))
    vel: tuple = (0.0, 0.0)

    @property
    def conf(self) -> float:
        return self.scores.get(self.cls, 0.0)

    @property
    def unsure(self) -> bool:
        """Weak or conflicting class evidence. This is the VLM trigger later."""
        s = sorted(self.scores.values(), reverse=True) + [0.0]
        return s[0] < 0.55 or (s[1] > 0.2 and s[1] > 0.5 * s[0])


class Tracker:
    """Greedy IoU tracker with class hysteresis.

    The class label is a running average of detector confidence per class, and it
    only switches when another class is clearly ahead (`margin`). One bad frame
    cannot turn a person into a pillar. People coast longer than other classes
    when the detector loses them, because losing a person is the unsafe error.
    """

    def __init__(self, iou_min=0.15, min_hits=2, decay=0.8, margin=0.15,
                 max_age=None):
        self.iou_min, self.min_hits, self.decay, self.margin = iou_min, min_hits, decay, margin
        self.max_age = max_age or {"person": 15, "pillar": 8}
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection], t: float) -> list[Track]:
        pairs = sorted(((iou(tr.bbox, d.bbox), i, j)
                        for i, tr in enumerate(self.tracks) for j, d in enumerate(dets)),
                       reverse=True)
        used_t, used_d = set(), set()
        for v, i, j in pairs:
            if v < self.iou_min:
                break
            if i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            self._match(self.tracks[i], dets[j], t)

        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks
                       if tr.misses <= self.max_age.get(tr.cls, 8)]

        for j, d in enumerate(dets):
            if j not in used_d:
                self.tracks.append(Track(self._next_id, d.bbox, {d.cls: d.conf}, d.cls, last_seen=t))
                self._next_id += 1

        return [tr for tr in self.tracks if tr.hits >= self.min_hits]

    def _match(self, tr: Track, d: Detection, t: float) -> None:
        tr.hits += 1
        tr.misses = 0
        tr.last_seen = t
        tr.bbox = tuple(0.4 * o + 0.6 * n for o, n in zip(tr.bbox, d.bbox))
        for c in set(tr.scores) | {d.cls}:
            tr.scores[c] = self.decay * tr.scores.get(c, 0.0) + (1 - self.decay) * (d.conf if c == d.cls else 0.0)
        best = max(tr.scores, key=tr.scores.get)
        if best != tr.cls and tr.scores[best] > tr.scores.get(tr.cls, 0.0) + self.margin:
            tr.cls = best


# ------------------------------------------------------------- localisation
def locate(bbox, cls, cam, lidar_angles, ranges):
    """(ccw bearing rel. to heading, range to object centre, source) or None.

    Primary: lidar range at the bearing of the box. Fallback: monocular
    ground-contact distance (needs the feet in view).
    """
    x1, y1, x2, y2 = bbox
    rel_right = math.atan(((x1 + x2) / 2 - cam.W / 2) / cam.f)
    bearing = -rel_right                                   # image right = clockwise
    half = max(math.atan((x2 - x1) * 0.25 / cam.f), math.radians(0.75))
    diff = np.angle(np.exp(1j * (lidar_angles - bearing)))
    sel = ranges[np.abs(diff) <= half]
    sel = sel[np.isfinite(sel)]
    r = CLASS_RADIUS.get(cls, 0.3)
    if sel.size:
        return bearing, float(sel.min()) + r, "lidar"
    below = y2 - cam.H / 2
    if below > 3 and y2 < cam.H - 1:
        dperp = cam.f * CAM_H / below
        return bearing, dperp / max(math.cos(rel_right), 0.2), "camera"
    return None


@dataclass
class Obstacle:
    id: int
    cls: str
    conf: float
    unsure: bool
    bearing: float        # rad, CCW from robot heading
    range: float          # m, to object centre
    x: float
    y: float
    vx: float
    vy: float
    coasting: bool        # detector lost it this frame; position is held
    source: str           # "lidar" (trusted) or "camera" (beyond lidar range, coarse)
    vel_valid: bool = True
    person_score: float = 0.0   # smoothed evidence that this is a person, even if another class won

    @property
    def reliable(self) -> bool:
        return self.source == "lidar"

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def moving(self) -> bool:
        return self.vel_valid and self.speed > 0.25


class Perception:
    def __init__(self, detector, camera, lidar):
        self.detector, self.cam, self.lidar = detector, camera, lidar
        self.tracker = Tracker()

    def update(self, frame, ranges, robot, t: float) -> list[Obstacle]:
        tracks = self.tracker.update(self.detector(frame), t)
        out = []
        for tr in tracks:
            coasting = tr.misses > 0
            if not coasting:
                loc = locate(tr.bbox, tr.cls, self.cam, self.lidar.rel_angles, ranges)
                if loc is not None:
                    b, rng, src = loc
                    ang = robot.theta + b
                    p = (robot.x + rng * math.cos(ang), robot.y + rng * math.sin(ang))
                    a = 0.5 if src == "lidar" else 0.15      # camera-only range is noisy: smooth harder
                    tr.pos = p if tr.pos is None else ((1 - a) * tr.pos[0] + a * p[0], (1 - a) * tr.pos[1] + a * p[1])
                    if src == "lidar":
                        tr.hist.append((t, *tr.pos))
                        if len(tr.hist) >= 2 and tr.hist[-1][0] - tr.hist[0][0] >= 0.2:
                            t0, x0, y0 = tr.hist[0]
                            tr.vel = ((tr.pos[0] - x0) / (t - t0), (tr.pos[1] - y0) / (t - t0))
                            tr.vel_ok = True
                    else:                                    # out of lidar range: velocity not trustworthy
                        tr.hist.clear()
                        tr.vel, tr.vel_ok = (0.0, 0.0), False
                    tr._last = (b, rng, src)
            if tr.pos is None or not hasattr(tr, "_last"):
                continue
            b, rng, src = tr._last
            out.append(Obstacle(tr.id, tr.cls, tr.conf, tr.unsure, b, rng, tr.pos[0], tr.pos[1],
                                tr.vel[0], tr.vel[1], coasting, src, getattr(tr, "vel_ok", False),
                                tr.scores.get("person", 0.0)))
        return out
