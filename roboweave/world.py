"""Simulated warehouse: static racks and pillars, walking people, one robot.

Units are metres and radians. The world is continuous (not a grid); the
semantic grid for planning is built later from sensor data, not from here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pillar:
    x: float
    y: float
    r: float = 0.30
    kind: str = "pillar"


@dataclass
class Rack:
    x: float  # min corner
    y: float
    w: float
    h: float
    kind: str = "rack"


@dataclass
class Person:
    x: float
    y: float
    route: list = field(default_factory=list)  # [(x, y), ...] looped
    speed: float = 0.8
    pause: float = 1.0
    r: float = 0.25
    kind: str = "person"
    style: dict | None = None   # optional colours: vest, pants, skin, helmet
    _i: int = 0
    _wait: float = 0.0

    def step(self, dt: float) -> None:
        if not self.route:
            return
        if self._wait > 0:
            self._wait -= dt
            return
        tx, ty = self.route[self._i]
        dx, dy = tx - self.x, ty - self.y
        d = math.hypot(dx, dy)
        move = self.speed * dt
        if d <= move:
            self.x, self.y = tx, ty
            self._i = (self._i + 1) % len(self.route)
            self._wait = self.pause
        else:
            self.x += dx / d * move
            self.y += dy / d * move


@dataclass
class Robot:
    x: float
    y: float
    theta: float = 0.0
    r: float = 0.35
    v: float = 0.0
    w: float = 0.0
    max_v: float = 1.2
    max_w: float = 2.0


def _circle_rect_hit(cx, cy, r, rk: Rack) -> bool:
    nx = min(max(cx, rk.x), rk.x + rk.w)
    ny = min(max(cy, rk.y), rk.y + rk.h)
    return (cx - nx) ** 2 + (cy - ny) ** 2 < r * r


class World:
    def __init__(self, width, height, racks, pillars, people, robot):
        self.width, self.height = width, height
        self.racks, self.pillars, self.people, self.robot = racks, pillars, people, robot
        self.time = 0.0
        self.collisions = 0       # distinct contact events (robot vs anything)
        self._in_contact = False
        self._build_segments()

    # ---- geometry used by the sensors -------------------------------------
    def _build_segments(self) -> None:
        """Wall segments as rows (x1, y1, x2, y2) plus a kind per row."""
        segs, kinds = [], []
        W, H = self.width, self.height
        for a, b in [((0, 0), (W, 0)), ((W, 0), (W, H)), ((W, H), (0, H)), ((0, H), (0, 0))]:
            segs.append((*a, *b)); kinds.append("wall")
        for rk in self.racks:
            x0, y0, x1, y1 = rk.x, rk.y, rk.x + rk.w, rk.y + rk.h
            for a, b in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                         ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
                segs.append((*a, *b)); kinds.append("rack")
        self.segments = np.array(segs, dtype=float)
        self.segment_kinds = kinds

    def circles(self):
        """Round obstacles as (x, y, r, kind). Pillars and people."""
        return [(p.x, p.y, p.r, p.kind) for p in self.pillars] + \
               [(p.x, p.y, p.r, p.kind) for p in self.people]

    # ---- dynamics ---------------------------------------------------------
    def _robot_collides(self, x, y) -> bool:
        r = self.robot.r
        if x - r < 0 or y - r < 0 or x + r > self.width or y + r > self.height:
            return True
        if any(_circle_rect_hit(x, y, r, rk) for rk in self.racks):
            return True
        return any((x - cx) ** 2 + (y - cy) ** 2 < (r + cr) ** 2
                   for cx, cy, cr, _ in self.circles())

    def step(self, dt: float, v_cmd: float = 0.0, w_cmd: float = 0.0) -> None:
        for p in self.people:
            p.step(dt)
        rb = self.robot
        rb.v = max(-rb.max_v, min(rb.max_v, v_cmd))
        rb.w = max(-rb.max_w, min(rb.max_w, w_cmd))
        th = rb.theta + rb.w * dt
        nx = rb.x + rb.v * math.cos(th) * dt
        ny = rb.y + rb.v * math.sin(th) * dt
        blocked = self._robot_collides(nx, ny)
        if blocked:
            # contact: robot does not move; count one event per contact
            if not self._in_contact:
                self.collisions += 1
            self._in_contact = True
        else:
            rb.x, rb.y = nx, ny
            self._in_contact = False
        rb.theta = (th + math.pi) % (2 * math.pi) - math.pi
        self.time += dt


def default_world() -> World:
    racks = [Rack(3, 2.0, 7, 0.8), Rack(3, 5.0, 7, 0.8), Rack(14, 2.0, 7, 0.8), Rack(14, 5.0, 7, 0.8),
             Rack(3, 8.5, 7, 0.8), Rack(3, 11.5, 7, 0.8), Rack(14, 8.5, 7, 0.8), Rack(14, 11.5, 7, 0.8)]
    pillars = [Pillar(11.2, 6.4), Pillar(12.8, 9.6)]
    people = [
        Person(12.0, 1.0, route=[(12.0, 1.0), (12.0, 13.0)], speed=0.8),
        Person(6.0, 7.4, route=[(6.0, 7.4), (22.0, 7.4)], speed=0.7, pause=2.0),
    ]
    robot = Robot(1.5, 7.0, 0.0)
    return World(24.0, 14.0, racks, pillars, people, robot)
