"""Semantic occupancy grid.

Geometry comes from lidar, class comes from the camera, and the safety margin is
set per class. This is the one place where "a person is not a pillar" turns into
numbers the planner can use.

Layers
  static map   racks and walls from the floor plan (known in advance, like a real
               warehouse map), inflated by robot radius + a small margin
  perceived    everything the lidar sees that the floor plan does not explain:
               people, pillars, anything else. A lidar hit that falls on a tracked
               obstacle takes that obstacle's class; unexplained hits are "unknown".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roboweave.perception import CLASS_RADIUS

# semantic labels (also used by the dashboard map)
FREE, STATIC_MAP, PILLAR, PERSON, UNKNOWN = 0, 1, 2, 3, 4

PERSON_ZONE = 2.0        # m from a person's centre: stop-and-yield zone
BOX_MARGIN = 0.25        # m of clearance beyond robot radius for ordinary obstacles
MEMORY_S = 20.0          # s an unexplained object stays on the map unless a lidar beam sees through it


@dataclass
class GridFrame:
    blocked: np.ndarray          # bool [row=y, col=x]
    semantic: np.ndarray         # int8 labels for display
    unexplained: np.ndarray      # (N, 2) world points the floor plan does not explain


def _offsets(r_cells: float):
    n = int(math.ceil(r_cells))
    return [(dy, dx) for dy in range(-n, n + 1) for dx in range(-n, n + 1)
            if dy * dy + dx * dx <= r_cells * r_cells]


def dilate(mask: np.ndarray, r_cells: float) -> np.ndarray:
    out = mask.copy()
    H, W = mask.shape
    for dy, dx in _offsets(r_cells):
        if dy == 0 and dx == 0:
            continue
        ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
        xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
        out[ys, xs] |= mask[yd, xd]
    return out


class SemanticGrid:
    def __init__(self, world, res=0.2, robot_r=0.35, static_margin=0.15):
        self.res, self.robot_r = res, robot_r
        self.nx, self.ny = int(math.ceil(world.width / res)), int(math.ceil(world.height / res))
        prior = np.zeros((self.ny, self.nx), bool)
        for rk in world.racks:
            i0, i1 = int((rk.x - 0.05) / res), int((rk.x + rk.w + 0.05) / res)
            j0, j1 = int((rk.y - 0.05) / res), int((rk.y + rk.h + 0.05) / res)
            prior[max(j0, 0):j1 + 1, max(i0, 0):i1 + 1] = True
        prior[0, :] = prior[-1, :] = True
        prior[:, 0] = prior[:, -1] = True
        self.prior = prior
        self.prior_tol = dilate(prior, 1.0)                 # tolerance for matching lidar hits
        self.static_blocked = dilate(prior, (robot_r + static_margin) / res)
        self.last_hit = np.full((self.ny, self.nx), -1e9)         # when each cell last returned an unexplained hit
        self._ds = np.arange(0.4, 12.2, res)                     # sample distances along a beam for free-space clearing

    # ---- helpers ------------------------------------------------------------
    def cell(self, x, y):
        return (min(max(int(y / self.res), 0), self.ny - 1), min(max(int(x / self.res), 0), self.nx - 1))

    def centre(self, row, col):
        return ((col + 0.5) * self.res, (row + 0.5) * self.res)

    def _disc(self, x, y, rad):
        """Boolean mask of cells whose centre is within rad of (x, y)."""
        res = self.res
        j0, j1 = max(int((y - rad) / res) - 1, 0), min(int((y + rad) / res) + 2, self.ny)
        i0, i1 = max(int((x - rad) / res) - 1, 0), min(int((x + rad) / res) + 2, self.nx)
        m = np.zeros((self.ny, self.nx), bool)
        if j0 >= j1 or i0 >= i1:
            return m
        yy, xx = np.mgrid[j0:j1, i0:i1]
        m[j0:j1, i0:i1] = ((xx + 0.5) * res - x) ** 2 + ((yy + 0.5) * res - y) ** 2 <= rad * rad
        return m

    def _clear_free_space(self, robot, ranges, rel_angles):
        """Cells a beam travelled through (stopping 0.3 m short of its hit) are free: forget them."""
        r = np.where(np.isfinite(ranges), ranges - 0.3, np.inf)
        ang = robot.theta + rel_angles
        ds = self._ds[None, :]
        ok = ds < r[:, None]
        ii = ((robot.x + ds * np.cos(ang)[:, None]) / self.res).astype(int)
        jj = ((robot.y + ds * np.sin(ang)[:, None]) / self.res).astype(int)
        ok &= (ii >= 0) & (ii < self.nx) & (jj >= 0) & (jj < self.ny)
        self.last_hit[jj[ok], ii[ok]] = -1e9

    # ---- per-frame build ------------------------------------------------------
    def build(self, robot, ranges, rel_angles, obstacles, use_classes=True, block_people=False, t=0.0) -> GridFrame:
        """use_classes=False builds the lidar-only grid: every non-map hit is a box.

        People get their 2 m zone in the semantic layer. By default the zone does NOT
        block planning: the decision layer yields (stops and waits) when the path
        enters it, instead of dodging in front of someone who is walking."""
        fin = np.isfinite(ranges)
        ang = robot.theta + rel_angles[fin]
        pts = np.stack([robot.x + ranges[fin] * np.cos(ang), robot.y + ranges[fin] * np.sin(ang)], axis=1)
        if len(pts):
            ci = np.clip((pts[:, 0] / self.res).astype(int), 0, self.nx - 1)
            cj = np.clip((pts[:, 1] / self.res).astype(int), 0, self.ny - 1)
            unexplained = pts[~self.prior_tol[cj, ci]]
        else:
            unexplained = pts

        blocked = self.static_blocked.copy()
        sem = np.zeros((self.ny, self.nx), np.int8)
        sem[self.prior] = STATIC_MAP

        keep = np.ones(len(unexplained), bool)
        classed = []
        if use_classes:
            for o in obstacles:
                if len(unexplained):
                    d = np.hypot(unexplained[:, 0] - o.x, unexplained[:, 1] - o.y)
                    keep &= d > CLASS_RADIUS.get(o.cls, 0.3) + 0.3
                classed.append(o)
        rest = unexplained[keep]

        # unknown: unexplained lidar hits, ordinary margin. Remembered for MEMORY_S so the robot does not
        # forget a pallet the moment it turns away; a beam passing through the spot clears it.
        self._clear_free_space(robot, ranges, rel_angles)
        if len(rest):
            hj = np.clip((rest[:, 1] / self.res).astype(int), 0, self.ny - 1)
            hi = np.clip((rest[:, 0] / self.res).astype(int), 0, self.nx - 1)
            self.last_hit[hj, hi] = t
        seen = (t - self.last_hit) < MEMORY_S
        if seen.any():
            zone = dilate(seen, (self.robot_r + BOX_MARGIN) / self.res)
            blocked |= zone
            sem[zone & (sem == FREE)] = UNKNOWN

        # classified obstacles: margin depends on class
        for o in classed:
            if o.cls == "person":
                zone, lab = self._disc(o.x, o.y, PERSON_ZONE), PERSON
            else:
                rad = CLASS_RADIUS.get(o.cls, 0.3) + self.robot_r + BOX_MARGIN
                zone, lab = self._disc(o.x, o.y, rad), PILLAR
            if o.cls != "person" or block_people:
                blocked |= zone
            sem[zone & (sem != STATIC_MAP)] = lab
        return GridFrame(blocked, sem, unexplained)
