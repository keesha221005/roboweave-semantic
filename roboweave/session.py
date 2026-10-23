"""One simulated run as an object: step it, snapshot it, draw its camera.

The server steps a Session in real time; the headless runner steps it flat out.
Same loop as roboweave.navigate (sensors -> perception -> decision -> motion).
"""
from __future__ import annotations

import io
import math
import os
from collections import deque

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from roboweave.decision import Navigator  # noqa: E402
# pyrefly: ignore[missing-import]
from roboweave.metrics import Metrics  # noqa: E402
from roboweave.perception import NoisyGTDetector, Perception, YoloDetector  # noqa: E402
from roboweave.scenarios import NAMES, build  # noqa: E402
from roboweave.sensors import Camera, Lidar  # noqa: E402
from roboweave.vlm import make_client  # noqa: E402

DT = 1 / 30
MODES = ("roboweave", "box", "stop_all")
_YOLO: dict = {}                      # weights path -> loaded detector (loading is slow)


def _detector(weights, seed=None):
    if not weights:
        return NoisyGTDetector(seed=1 if seed is None else seed)
    if weights not in _YOLO:
        _YOLO[weights] = YoloDetector(weights)
    return _YOLO[weights]


class Session:
    def __init__(self, scenario="crossing", mode="roboweave", weights=None, vlm="none",
                 vlm_model=None, vlm_timeout=4.0, timeout=90.0, seed=None):
        assert scenario in NAMES and mode in MODES
        self.scenario, self.mode, self.weights, self.vlm_kind = scenario, mode, weights, vlm
        self.timeout = timeout
        self.world, self.goal = build(scenario, seed)
        self.lidar, self.cam = Lidar(seed=3 if seed is None else seed), Camera()
        self.det = _detector(weights, seed)
        self.fake = self.det if isinstance(self.det, NoisyGTDetector) else None
        self.perc = Perception(self.det, self.cam, self.lidar)
        vlm_mode = vlm if mode == "roboweave" else "none"
        self.nav = Navigator(self.world, mode, self.lidar, make_client(vlm_mode, vlm_model, self.world, 1.0, vlm_timeout))
        self.t, self.done_t = 0.0, None
        self.finished = False
        self.cmd = None
        self.obs: list = []
        self.frame = None
        self.state_time: dict = {}
        self.trace: deque = deque(maxlen=240)          # (t, speed, nearest person gap)
        self._next_trace = 0.0
        self.metrics = Metrics(DT)
        self.frame, _ = self.cam.render(self.world)       # something to show before the first step

    # ------------------------------------------------------------------ run
    def step(self):
        if self.finished:
            return
        w = self.world
        ranges = self.lidar.scan(w)
        self.frame, gt = self.cam.render(w)
        if self.fake:
            self.fake.gt = gt
        self.obs = self.perc.update(self.frame, ranges, w.robot, self.t)
        self.cmd = self.nav.step(self.t, w.robot, ranges, self.obs, self.goal, self.frame)
        c = self.cmd
        self.metrics.update(w, c, self.obs, gt, len(self.nav.log))
        self.state_time[c.state] = self.state_time.get(c.state, 0.0) + DT
        if c.state == "GO" and c.slow < 1.0:
            self.state_time["slowed"] = self.state_time.get("slowed", 0.0) + DT
        if self.t >= self._next_trace:
            self._next_trace = self.t + 0.2
            self.trace.append((round(self.t, 1), round(c.v, 2), self.nearest_person()))
        w.step(DT, c.v, c.w)
        self.t += DT
        if c.state == "DONE":
            self.done_t, self.finished = self.t, True
        elif self.t >= self.timeout:
            self.finished = True

    def nearest_person(self):
        r = self.world.robot
        gaps = [math.hypot(p.x - r.x, p.y - r.y) - p.r - r.r for p in self.world.people]
        return round(min(gaps), 2) if gaps else None

    # ------------------------------------------------------------- outputs
    def result(self):
        w = self.world
        return {"scenario": self.scenario, "mode": self.mode, "time": self.done_t,
                "collisions": w.collisions,
                "min_gap": None if math.isinf(w.min_person_gap) else round(w.min_person_gap, 2),
                "state_time": {k: round(v, 1) for k, v in self.state_time.items()},
                "log": list(self.nav.log), "metrics": self.metrics.summary()}

    def snapshot(self):
        w, r = self.world, self.world.robot
        from roboweave.grid import PERSON_ZONE
        return {
            "t": round(self.t, 1), "scenario": self.scenario, "mode": self.mode,
            "vlm": self.vlm_kind if self.mode == "roboweave" else "none",
            "detector": "YOLOv8 (fine-tuned)" if self.weights else "simulated detector (no weights)",
            "finished": self.finished, "done_t": self.done_t,
            "state": self.cmd.state if self.cmd else "GO", "speed": round(self.cmd.v, 2) if self.cmd else 0.0,
            "slow": bool(self.cmd and self.cmd.slow < 1.0),
            "reason": self.cmd.reason if self.cmd else "",
            "collisions": w.collisions,
            "min_gap": None if math.isinf(w.min_person_gap) else round(w.min_person_gap, 2),
            "nearest_person": self.nearest_person(),
            "dist_to_goal": round(math.hypot(self.goal[0] - r.x, self.goal[1] - r.y), 1),
            "goal": list(self.goal), "yield_zone": PERSON_ZONE,
            "robot": {"x": round(r.x, 2), "y": round(r.y, 2), "theta": round(r.theta, 3), "r": r.r},
            "world": {"width": w.width, "height": w.height,
                      "racks": [[k.x, k.y, k.w, k.h] for k in w.racks],
                      "crates": [[k.x, k.y, k.w, k.h] for k in (w.crates or [])],
                      "pillars": [[p.x, p.y, p.r] for p in w.pillars]},
            "people": [[round(p.x, 2), round(p.y, 2), p.r] for p in w.people],
            "path": [[round(x, 2), round(y, 2)] for x, y in self.nav.path],
            "perceived": [{"id": o.id, "cls": o.cls, "conf": round(o.conf, 2), "unsure": bool(o.unsure),
                           "x": round(o.x, 2), "y": round(o.y, 2), "range": round(o.range, 1)} for o in self.obs],
            "state_time": {k: round(v, 1) for k, v in self.state_time.items()},
            "trace": list(self.trace),
            "log": [[a, b] for a, b in self.nav.log[-60:]],
        }

    def jpeg(self, width=480):
        """Camera frame with the tracker's boxes, as JPEG bytes."""
        from PIL import Image, ImageDraw
        im = Image.fromarray(self.frame)
        d = ImageDraw.Draw(im)
        for tr in self.perc.tracker.tracks:
            if tr.hits >= self.perc.tracker.min_hits:
                col = (0, 255, 0) if tr.cls == "person" else (80, 160, 255)
                d.rectangle(tr.bbox, outline=col, width=3)
                d.text((tr.bbox[0] + 3, max(0, tr.bbox[1] - 12)), f"{tr.cls} {tr.conf:.2f}", fill=col)
        im = im.resize((width, int(width * im.height / im.width)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return buf.getvalue()


def run_headless(scenario, mode, weights=None, vlm="none", vlm_model=None, timeout=60.0, seed=None):
    s = Session(scenario, mode, weights, vlm, vlm_model, timeout=timeout, seed=seed)
    while not s.finished:
        s.step()
    return s.result()