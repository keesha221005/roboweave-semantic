"""Per-run measurements, computed from simulator ground truth (never from what the robot believes).

zone compliance  While a real person is inside 2.0 m of the robot centre and in front of it, for
                 at least 0.5 s (reaction grace), the robot should not be moving (v <= 0.05).
                 compliance = share of those steps where it was stopped.
near misses      Separate episodes where the surface-to-surface gap to a person fell under 0.5 m.
false yield      Seconds spent in YIELD while no real person was within 3.6 m (zone 2.0 m + 1.5 m
                 path lookahead + margin). Wasted time caused by false alarms.
stuck            Seconds in STOP or WAIT (not yielding): blocked with nothing to wait for.
person recall    Frames where a real person was visible within 8 m and the tracker had a
                 person-class obstacle within 1.0 m of where they actually stood.
false person     Share of person-class obstacle frames with no real person within 1.0 m.
logging          State changes that produced a decision-log line, out of all state changes.
"""
from __future__ import annotations

import math

ZONE = 2.0
GRACE = 0.5
NEAR_MISS = 0.5
PHANTOM_RADIUS = 3.6
MOVING = 0.05


class Metrics:
    def __init__(self, dt):
        self.dt = dt
        self.zone_steps = self.zone_moving = 0
        self._in_zone_s = 0.0
        self.near_misses, self._in_near = 0, False
        self.false_yield_s = 0.0
        self.stuck_s = 0.0
        self.vis = self.vis_hit = 0
        self.pers_obs = self.pers_false = 0
        self.changes = self.logged = 0
        self._state, self._log_len = "GO", 0

    def update(self, world, cmd, obs, gt, log_len):
        r = world.robot
        # --- people geometry (ground truth)
        d_front = []
        nearest = math.inf
        gap = math.inf
        for p in world.people:
            d = math.hypot(p.x - r.x, p.y - r.y)
            nearest, gap = min(nearest, d), min(gap, d - p.r - r.r)
            rel = math.atan2(math.sin(math.atan2(p.y - r.y, p.x - r.x) - r.theta),
                             math.cos(math.atan2(p.y - r.y, p.x - r.x) - r.theta))
            if d < ZONE and (abs(rel) < 1.9 or d < 1.0):
                d_front.append(d)
        # --- yield compliance
        if d_front:
            self._in_zone_s += self.dt
            if self._in_zone_s >= GRACE:
                self.zone_steps += 1
                self.zone_moving += cmd.v > MOVING
        else:
            self._in_zone_s = 0.0
        # --- near misses (episodes)
        near = gap < NEAR_MISS
        if near and not self._in_near:
            self.near_misses += 1
        self._in_near = near
        # --- wasted / stuck time
        if cmd.state == "YIELD" and nearest > PHANTOM_RADIUS:
            self.false_yield_s += self.dt
        if cmd.state in ("STOP", "WAIT"):
            self.stuck_s += self.dt
        # --- detection quality
        people_xy = [(p.x, p.y) for p in world.people]
        for g in gt:
            if g["cls"] == "person" and g["dist"] <= 8.0:
                a = r.theta - g["bearing"]          # camera bearing is clockwise-positive
                x, y = r.x + g["dist"] * math.cos(a), r.y + g["dist"] * math.sin(a)
                self.vis += 1
                self.vis_hit += any(o.cls == "person" and math.hypot(o.x - x, o.y - y) < 1.0 for o in obs)
        for o in obs:
            if o.cls == "person":
                self.pers_obs += 1
                self.pers_false += not any(math.hypot(o.x - x, o.y - y) < 1.0 for x, y in people_xy)
        # --- explainability
        if cmd.state != self._state:
            self.changes += 1
            self.logged += log_len > self._log_len
            self._state = cmd.state
        self._log_len = log_len

    def summary(self):
        return {
            "zone_steps": self.zone_steps, "zone_moving": self.zone_moving,
            "near_misses": self.near_misses,
            "false_yield_s": round(self.false_yield_s, 1), "stuck_s": round(self.stuck_s, 1),
            "vis": self.vis, "vis_hit": self.vis_hit,
            "pers_obs": self.pers_obs, "pers_false": self.pers_false,
            "changes": self.changes, "logged": self.logged,
        }