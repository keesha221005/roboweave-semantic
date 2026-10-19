"""Simulated lidar (geometry only) and camera (appearance only).

The lidar returns ranges and nothing else: it cannot tell a person from a
pillar. The camera returns an RGB frame plus ground-truth boxes, which are
used later to score the detector and never fed to the robot.
"""
from __future__ import annotations

import math

import numpy as np


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def raycast_segments(ox, oy, angles, segs):
    """Nearest hit per ray against wall segments. Returns (dist, seg_index)."""
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]       # (N,1)
    qx, qy = segs[None, :, 0], segs[None, :, 1]                      # (1,M)
    sx, sy = segs[None, :, 2] - qx, segs[None, :, 3] - qy
    den = _cross(dx, dy, sx, sy)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qx - ox, qy - oy, sx, sy) / den
        u = _cross(qx - ox, qy - oy, dx, dy) / den
    ok = (np.abs(den) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    idx = np.argmin(t, axis=1)
    return t[np.arange(len(angles)), idx], idx


def raycast_circles(ox, oy, angles, circles):
    """Nearest hit per ray against circles [(x, y, r, kind)]."""
    best = np.full(len(angles), np.inf)
    dx, dy = np.cos(angles), np.sin(angles)
    for cx, cy, r, _ in circles:
        px, py = ox - cx, oy - cy
        b = dx * px + dy * py
        c = px * px + py * py - r * r
        disc = b * b - c
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        t = np.where((disc >= 0) & (t >= 0), t, np.inf)
        best = np.minimum(best, t)
    return best


class Lidar:
    def __init__(self, n_beams=360, max_range=12.0, noise=0.02, seed=0):
        self.n, self.max_range, self.noise = n_beams, max_range, noise
        self.rng = np.random.default_rng(seed)
        self.rel_angles = np.linspace(-math.pi, math.pi, n_beams, endpoint=False)

    def scan(self, world):
        rb = world.robot
        ang = rb.theta + self.rel_angles
        d_seg, _ = raycast_segments(rb.x, rb.y, ang, world.segments)
        d_cir = raycast_circles(rb.x, rb.y, ang, world.circles())
        d = np.minimum(d_seg, d_cir)
        d = d + self.rng.normal(0, self.noise, d.shape)
        d = np.where(d > self.max_range, np.inf, d)
        return d                      # inf = no return; same length as rel_angles


# --------------------------------------------------------------------------
# Camera: pseudo-3D first-person render. Walls/racks are ray-cast per column,
# people and pillars are drawn as upright billboard sprites, far to near.
# --------------------------------------------------------------------------
CAM_H = 0.8        # camera height (m)
WALL_H = 3.0
PERSON_H = 1.7
PILLAR_H = 3.0

KIND_COLOR = {"wall": (120, 130, 145), "rack": (196, 120, 50)}


class Camera:
    def __init__(self, width=640, height=360, fov_deg=90.0, max_range=14.0):
        import pygame  # imported lazily so lidar-only use needs no pygame
        self.pg = pygame
        self.W, self.H = width, height
        self.fov = math.radians(fov_deg)
        self.f = (width / 2) / math.tan(self.fov / 2)
        self.max_range = max_range
        self.col_rel = np.arctan((np.arange(width) - width / 2 + 0.5) / self.f)

    def render(self, world):
        """Return (frame HxWx3 uint8, ground_truth list of dicts)."""
        pg, W, H, f = self.pg, self.W, self.H, self.f
        surf = pg.Surface((W, H))
        hor = H // 2
        surf.fill((28, 32, 40), (0, 0, W, hor))                       # ceiling
        for y in range(hor, H):                                       # floor gradient
            k = (y - hor) / (H - hor)
            surf.fill((int(52 + 40 * k),) * 3, (0, y, W, 1))

        rb = world.robot
        ang = rb.theta - self.col_rel   # +x of image = robot's right (clockwise)
        d, idx = raycast_segments(rb.x, rb.y, ang, world.segments)
        zbuf = d * np.cos(self.col_rel)
        for c in range(W):
            if not np.isfinite(d[c]) or zbuf[c] > self.max_range:
                continue
            top = int(hor - f * (WALL_H - CAM_H) / zbuf[c])
            bot = int(hor + f * CAM_H / zbuf[c])
            base = KIND_COLOR[world.segment_kinds[idx[c]]]
            shade = max(0.35, 1.0 - zbuf[c] / (self.max_range * 1.1))
            vertical = abs(world.segments[idx[c], 2] - world.segments[idx[c], 0]) < 1e-9
            shade *= 0.8 if vertical else 1.0
            surf.fill(tuple(int(v * shade) for v in base), (c, top, 1, bot - top))

        # sprites, far to near
        items = []
        for ob in list(world.pillars) + list(world.people):
            cx, cy, r, kind = ob.x, ob.y, ob.r, ob.kind
            dx, dy = cx - rb.x, cy - rb.y
            dist = math.hypot(dx, dy)
            diff = rb.theta - math.atan2(dy, dx)            # positive = target to the right
            rel = math.atan2(math.sin(diff), math.cos(diff))
            items.append((dist, rel, r, kind, ob))
        items.sort(key=lambda t: -t[0])

        gt = []
        for dist, rel, r, kind, ob in items:
            if abs(rel) > self.fov / 2 + 0.2 or dist > self.max_range:
                continue
            dperp = dist * math.cos(rel)
            if dperp < 0.3:
                continue
            xc = W / 2 + f * math.tan(rel)
            wpx = f * (2 * r) / dperp
            h_m = PERSON_H if kind == "person" else PILLAR_H
            top = hor - f * (h_m - CAM_H) / dperp
            bot = hor + f * CAM_H / dperp
            col = int(min(max(xc, 0), W - 1))
            if dperp > zbuf[col] + r:          # hidden behind a wall/rack
                continue
            rect = (xc - wpx / 2, top, wpx, bot - top)
            shade = max(0.4, 1.0 - dperp / (self.max_range * 1.2))
            if kind == "person":
                self._draw_person(surf, rect, shade, getattr(ob, "style", None))
            else:
                self._draw_pillar(surf, rect, shade)
            x1, y1 = max(0, rect[0]), max(0, rect[1])
            x2, y2 = min(W, rect[0] + rect[2]), min(H, rect[1] + rect[3])
            if x2 - x1 > 2 and y2 - y1 > 2:
                gt.append({"cls": kind, "bbox": (x1, y1, x2, y2), "dist": dist, "bearing": rel})

        # gt is far-to-near; drop boxes mostly covered by a nearer object
        def covered(a, b):
            iw = min(a[2], b[2]) - max(a[0], b[0])
            ih = min(a[3], b[3]) - max(a[1], b[1])
            return iw > 0 and ih > 0 and iw * ih > 0.6 * (a[2] - a[0]) * (a[3] - a[1])
        gt = [g for i, g in enumerate(gt)
              if not any(covered(g["bbox"], h["bbox"]) for h in gt[i + 1:])]

        frame = np.transpose(pg.surfarray.array3d(surf), (1, 0, 2)).copy()
        return frame, gt

    def _draw_person(self, surf, rect, s, style=None):
        pg = self.pg
        x, y, w, h = rect
        sc = lambda c: tuple(int(v * s) for v in c)
        st = {"vest": (240, 190, 20), "pants": (40, 40, 70), "skin": (225, 180, 150), "helmet": (250, 250, 250)}
        st.update(style or {})
        head = h * 0.13
        pg.draw.rect(surf, sc(st["pants"]), (x + w * 0.18, y + h * 0.55, w * 0.28, h * 0.45))   # legs
        pg.draw.rect(surf, sc(st["pants"]), (x + w * 0.54, y + h * 0.55, w * 0.28, h * 0.45))
        pg.draw.rect(surf, sc(st["vest"]), (x + w * 0.1, y + h * 0.2, w * 0.8, h * 0.38))    # hi-vis vest
        pg.draw.rect(surf, sc((200, 200, 200)), (x + w * 0.1, y + h * 0.34, w * 0.8, h * 0.04))  # stripe
        pg.draw.circle(surf, sc(st["skin"]), (x + w / 2, y + head), max(2, head))           # head
        pg.draw.rect(surf, sc(st["helmet"]), (x + w * 0.32, y, w * 0.36, head * 0.9))         # helmet

    def _draw_pillar(self, surf, rect, s):
        pg = self.pg
        x, y, w, h = rect
        n = max(1, int(w))
        for i in range(n):                         # cylinder shading across width
            k = 0.65 + 0.35 * math.sin(math.pi * (i + 0.5) / n)
            c = tuple(int(v * s * k) for v in (150, 150, 155))
            pg.draw.rect(surf, c, (x + i * w / n, y, w / n + 1, h))
        pg.draw.rect(surf, tuple(int(v * s) for v in (200, 170, 40)), (x, y + h * 0.9, w, h * 0.1))  # hazard base
