"""Decision rules and motion control.

Three policies, same map, same scenarios, so they can be compared fairly:

  roboweave  class-aware. Person -> stop and yield inside a 2 m zone. Pillar or
             other static object -> A* routes around it, no stop. Unsure
             detection or unexplained object -> slow down and ask the VLM.
  stop_all   lidar only, stops for every obstacle ahead and waits for it to leave.
             Does not route around anything that is not on the floor plan.
  box        lidar only, treats everything as a box: routes around it with the
             ordinary margin. No stop-and-yield for people.

Safety rules that do not depend on the VLM:
  * When the evidence is weak but there is any sign of "person", treat it as a person.
  * The VLM can only make the robot MORE careful about a person, or relax a
    weak, already-unsure label. It never gates motion: if it is slow or down,
    the robot just stays slowed.

Every decision is logged with its reason in plain English.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from roboweave.grid import PERSON_ZONE, SemanticGrid
from roboweave.perception import CLASS_RADIUS
from roboweave.planner import astar, shorten

V_MAX = 1.0
W_MAX = 1.8
STOP_DIST = {"roboweave": 0.15, "box": 0.15, "stop_all": 1.5}   # m from the robot BODY: lidar stop / emergency distance
STOP_HYST = 0.3              # m: once stopped, stay stopped until the way is this much clearer
REPLAN_EVERY = 0.4           # s
PERSON_MEMORY = 2.0          # s: keep yielding to a person briefly after they leave the camera view
YIELD_LOOKAHEAD = 1.5        # m of planned path checked ahead of the robot (about the braking distance)
UNSURE_RANGE = 6.0           # m: only unsure objects this close matter for slowing down
UNSURE_SLOW = 0.5            # speed factor while a scene is unclear
PERSON_EVIDENCE = 0.25       # weak-label objects with at least this much "person" evidence are treated as people
STILL_WAIT = 8.0             # s of yielding to a person who has not moved before we pass them slowly
STILL_MOVE = 0.4             # m: less movement than this over STILL_WAIT counts as standing still
STILL_RELEASE = 0.7          # m: a passed person who moves this far from where they stood is a person again
UNCLEAR_FRAMES = 15          # an unexplained lidar object must persist this long before we call the scene unclear
UNCLEAR_RANGE = (0.8, 6.0)   # m window for unexplained objects dead ahead


@dataclass
class Command:
    v: float
    w: float
    state: str               # GO, YIELD, WAIT, STOP, DONE
    reason: str
    slow: float = 1.0        # <1 while the scene is unclear


class Navigator:
    def __init__(self, world, mode="roboweave", lidar=None, vlm=None):
        assert mode in STOP_DIST
        self.mode, self.lidar, self.vlm = mode, lidar, vlm
        self.grid = SemanticGrid(world)
        self.path: list = []                 # world waypoints
        self.cells: list = []                # same path as grid cells (for display)
        self.goal = None
        self.state = "GO"
        self.log: list[tuple[float, str]] = []
        self.frame = None
        self._last_plan = -1e9
        self._seen_static: set = set()
        self._people: dict = {}              # key -> (x, y, last seen t): short memory of people
        self._override: dict = {}            # track id -> "person" | "pillar" decided by the VLM
        self._asked: dict = {}               # scene key -> (x, y) we asked the VLM about
        self._vlm_people: set = set()        # cluster keys the VLM called a person
        self._unclear_frames = 0
        self._cluster = None
        self._tries: dict = {}               # scene key -> VLM asks so far (retry at most once after a timeout)
        self._last_cluster_ask = -1e9
        self._vlm_err_said = False
        self._no_vlm_said = False
        self._yield_since = None             # (t, {track id: (x, y)}) when the current yield began
        self._passing: dict = {}             # track id -> (x, y) of a motionless "person" we pass slowly

    # ------------------------------------------------------------------ log
    def say(self, t, text):
        self.log.append((round(t, 1), text))

    def _set_state(self, t, new, msg):
        if new != self.state:
            self.say(t, msg)
            self.state = new

    # ----------------------------------------------------------------- main
    def step(self, t, robot, ranges, obstacles, goal, frame=None) -> Command:
        if self.goal != goal:
            self.goal, self.path, self._last_plan = goal, [], -1e9
        if math.hypot(goal[0] - robot.x, goal[1] - robot.y) < 0.35:
            self._set_state(t, "DONE", "Goal reached")
            return Command(0.0, 0.0, "DONE", "goal reached")

        smart = self.mode == "roboweave"
        if smart:
            obstacles = [self._effective(o) for o in obstacles]
        for oid, (px, py) in list(self._passing.items()):          # a passed person who starts moving is a person again
            for o in obstacles:
                if o.id == oid and math.hypot(o.x - px, o.y - py) > STILL_RELEASE:
                    self._override.pop(oid, None)
                    del self._passing[oid]
                    self.say(t, "Person moving again, yielding")
                    break
        gf = self.grid.build(robot, ranges, self.lidar.rel_angles, obstacles, use_classes=smart, t=t)
        self.frame = gf
        if smart:
            self._track_unclear(robot, gf, obstacles)
            self._vlm_poll(t)

        # ---- 1. lidar stop (baselines) / emergency stop (all) for anything the map does not explain, dead ahead
        if len(gf.unexplained):
            d = gf.unexplained - np.array([robot.x, robot.y])
            dist = np.hypot(d[:, 0], d[:, 1])
            rel = np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - robot.theta)))
            lim = STOP_DIST[self.mode] + robot.r + (STOP_HYST if self.state == "STOP" else 0.0)
            ahead = (np.abs(rel) < math.radians(40)) & (dist < lim)
            if smart and obstacles:            # hits that belong to a tracked obstacle are handled by the planner / yield rule
                for o in obstacles:
                    ahead &= np.hypot(gf.unexplained[:, 0] - o.x, gf.unexplained[:, 1] - o.y) > 0.6
            if ahead.any():
                msg = "Obstacle ahead, stopping" if self.mode == "stop_all" else "Emergency stop, object very close"
                self._set_state(t, "STOP", msg)
                return Command(0.0, 0.0, "STOP", "lidar return inside stop distance")

        # ---- 2. plan (people are not planned around; see the yield rule)
        if not self.path or t - self._last_plan >= REPLAN_EVERY:
            self._plan(t, robot, gf, goal)
        if not self.path:
            self._set_state(t, "WAIT", "No clear path, waiting")
            return Command(0.0, 0.0, "WAIT", "no path")

        # ---- 3. person yield (class-aware only): stop if a person is in the way, wait until they leave
        if smart:
            for o in obstacles:
                if o.cls == "person":
                    self._people[("track", o.id)] = (o.x, o.y, t)
            if self._cluster:
                for key in self._vlm_people:                  # the VLM said the unexplained object is a person
                    self._people[("vlm", key)] = (self._cluster[0], self._cluster[1], t)
            self._people = {k: v for k, v in self._people.items() if t - v[2] <= PERSON_MEMORY}
            radius = PERSON_ZONE + (0.2 if self.state == "YIELD" else 0.0)     # hysteresis on release
            worst = self._person_conflict(robot, radius)
            if worst is None:
                self._yield_since = None
            else:
                tracks = {k[1]: (v[0], v[1]) for k, v in self._people.items() if k[0] == "track"}
                if self._yield_since is None:
                    self._yield_since = (t, tracks)
                elif t - self._yield_since[0] >= STILL_WAIT and tracks:
                    t0, first = self._yield_since
                    still = [i for i, (x, y) in tracks.items() if i in first and math.hypot(x - first[i][0], y - first[i][1]) < STILL_MOVE]
                    if still and len(still) == len(tracks):
                        for i in still:
                            self._override[i] = "pillar"
                            self._passing[i] = tracks[i]
                            self._people = {k: v for k, v in self._people.items() if k != ("track", i)}
                        self._yield_since = None
                        self.path, self._last_plan = [], -1e9
                        self.say(t, f"Person standing still for {STILL_WAIT:g} s, passing slowly")
                        return Command(0.0, 0.0, "WAIT", "replanning around a motionless person")
                    self._yield_since = (t, tracks)      # somebody moved: restart the clock
            if worst is not None:
                self._set_state(t, "YIELD", f"Person detected, yielding {PERSON_ZONE:g} m (nearest {worst:.1f} m)")
                return Command(0.0, 0.0, "YIELD", "person inside the yield zone")

        if self.state in ("YIELD", "WAIT", "STOP"):
            self._set_state(t, "GO", "Path clear, resuming speed")
        elif self.state == "DONE":
            self.state = "GO"

        # ---- 4. explain static objects; slow down and ask the VLM when the scene is unclear
        slow = UNSURE_SLOW if any(math.hypot(x - robot.x, y - robot.y) < 3.0 for x, y in self._passing.values()) else 1.0
        if smart:
            for o in obstacles:
                if o.cls != "person" and not o.unsure and o.id not in self._seen_static \
                        and abs(o.bearing) < 0.6 and o.range < 4.0:
                    self._seen_static.add(o.id)
                    self.say(t, f"{o.cls.title()} ahead, routing around ({o.range:.1f} m)")
                if o.unsure and o.range < UNSURE_RANGE:
                    slow = UNSURE_SLOW
                    self._ask(t, ("track", o.id), frame, (o.x, o.y))
            if self._unclear_frames >= UNCLEAR_FRAMES and self._cluster:
                slow = UNSURE_SLOW
                cx, cy = self._cluster[0], self._cluster[1]
                self._ask(t, ("cluster", int(cx // 3), int(cy // 3)), frame, (cx, cy))
        return self._drive(robot, goal, slow)

    # ------------------------------------------------------- weak labels / VLM
    def _effective(self, o):
        """Class the robot acts on. Weak labels with any person evidence count as people."""
        cls = self._override.get(o.id)
        if cls is None and o.cls != "person" and o.unsure and o.person_score >= PERSON_EVIDENCE:
            cls = "person"
        return replace(o, cls=cls) if cls and cls != o.cls else o

    def _track_unclear(self, robot, gf, obstacles):
        """Unexplained lidar object dead ahead that no tracked obstacle accounts for."""
        self._cluster = None
        pts = gf.unexplained
        if len(pts):
            d = pts - np.array([robot.x, robot.y])
            dist = np.hypot(d[:, 0], d[:, 1])
            rel = np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - robot.theta)))
            m = (np.abs(rel) < math.radians(35)) & (dist > UNCLEAR_RANGE[0]) & (dist < UNCLEAR_RANGE[1])
            for o in obstacles:
                m &= np.hypot(pts[:, 0] - o.x, pts[:, 1] - o.y) > CLASS_RADIUS.get(o.cls, 0.3) + 0.3
            if m.sum() >= 5:
                self._cluster = (float(pts[m, 0].mean()), float(pts[m, 1].mean()), int(m.sum()))
        self._unclear_frames = self._unclear_frames + 1 if self._cluster else max(0, self._unclear_frames - 2)

    def _ask(self, t, key, frame, where):
        if key in self._asked or self._tries.get(key, 0) >= 2:
            return
        if key[0] == "cluster" and t - self._last_cluster_ask < 10.0:
            return                                    # one VLM question per unexplained object, not one per step
        if self.vlm is None:
            self._asked[key] = where
            if not self._no_vlm_said:
                self._no_vlm_said = True
                self.say(t, "Scene unclear, no VLM available, staying cautious")
            return
        if frame is not None and self.vlm.submit(t, key, frame, {"x": where[0], "y": where[1]}):
            self._asked[key] = where
            self._tries[key] = self._tries.get(key, 0) + 1
            if key[0] == "cluster":
                self._last_cluster_ask = t
            self.say(t, "Scene unclear, asking VLM")

    def _vlm_poll(self, t):
        if self.vlm is None:
            return
        for key, adv in self.vlm.poll(t):
            if adv.timeout:
                err = (adv.raw or "").strip()
                if err and not self._vlm_err_said:
                    self._vlm_err_said = True
                    self.say(t, f"VLM error: {err[:90]}")
                self.say(t, "VLM unavailable, reducing speed" if err else "VLM slow, reducing speed")
                self._asked.pop(key, None)           # allow one retry later
                continue
            self.say(t, f"VLM says: {adv.text}")
            if key[0] == "track":
                if adv.object == "person":
                    self._override[key[1]] = "person"
                elif adv.object == "static":
                    self._override[key[1]] = "pillar"   # static, so route around instead of yielding
            elif adv.object == "person":
                self._vlm_people.add(key[1:])

    def _person_conflict(self, robot, radius):
        """Nearest remembered person if one is inside `radius` of the robot (and not behind it)
        or of the next YIELD_LOOKAHEAD metres of the planned path; otherwise None."""
        pts, left = [(robot.x, robot.y)], YIELD_LOOKAHEAD
        prev = pts[0]
        for wp in self.path:
            seg = math.hypot(wp[0] - prev[0], wp[1] - prev[1])
            n = max(int(min(seg, left) / 0.1), 1)
            f = min(seg, left) / seg if seg > 0 else 0.0
            for k in range(1, n + 1):
                pts.append((prev[0] + (wp[0] - prev[0]) * f * k / n, prev[1] + (wp[1] - prev[1]) * f * k / n))
            left -= seg
            if left <= 0:
                break
            prev = wp
        worst = None
        for x, y, _ in self._people.values():
            d_robot = math.hypot(x - robot.x, y - robot.y)
            rel = math.atan2(math.sin(math.atan2(y - robot.y, x - robot.x) - robot.theta),
                             math.cos(math.atan2(y - robot.y, x - robot.x) - robot.theta))
            d_path = min(math.hypot(x - px, y - py) for px, py in pts)
            near_me = d_robot < radius and (abs(rel) < 1.9 or d_robot < 1.0)
            if near_me or d_path < radius:
                worst = d_robot if worst is None else min(worst, d_robot)
        return worst

    # ------------------------------------------------------------------ plan
    def _plan(self, t, robot, gf, goal):
        self._last_plan = t
        g = self.grid
        blocked = gf.blocked
        if self.mode == "stop_all":
            blocked = g.static_blocked            # floor plan only: it will not route around what it stops for
        cells = astar(blocked, g.cell(robot.x, robot.y), g.cell(*goal))
        if cells is None:
            self.path, self.cells = [], []
            return
        cells = shorten(blocked, cells)
        self.cells = cells
        self.path = [g.centre(r, c) for r, c in cells][1:]
        if self.path:
            self.path[-1] = tuple(goal) if math.hypot(self.path[-1][0] - goal[0], self.path[-1][1] - goal[1]) < 0.4 else self.path[-1]

    # --------------------------------------------------------------- control
    def _drive(self, robot, goal, slow) -> Command:
        while len(self.path) > 1 and math.hypot(self.path[0][0] - robot.x, self.path[0][1] - robot.y) < 0.45:
            self.path.pop(0)
        tx, ty = self.path[0]
        err = math.atan2(math.sin(math.atan2(ty - robot.y, tx - robot.x) - robot.theta),
                         math.cos(math.atan2(ty - robot.y, tx - robot.x) - robot.theta))
        w = max(-W_MAX, min(W_MAX, 2.5 * err))
        dgoal = math.hypot(goal[0] - robot.x, goal[1] - robot.y)
        v = 0.0 if abs(err) > 0.9 else V_MAX * max(0.0, math.cos(err)) * slow
        v = min(v, 0.5 * dgoal + 0.15)
        return Command(v, w, "GO", "following path", slow)