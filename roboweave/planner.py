"""A* on the semantic grid (8-connected, no corner cutting) plus path shortening."""
from __future__ import annotations

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def _octile(a, b):
    dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def _nearest_free(blocked, cell, max_r=4):
    if not blocked[cell]:
        return cell
    H, W = blocked.shape
    best = None
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y, x = cell[0] + dy, cell[1] + dx
                if 0 <= y < H and 0 <= x < W and not blocked[y, x]:
                    d = dy * dy + dx * dx
                    if best is None or d < best[0]:
                        best = (d, (y, x))
        if best:
            return best[1]
    return None


def astar(blocked, start, goal):
    """Return a list of (row, col) cells from start to goal, or None."""
    H, W = blocked.shape
    goal = _nearest_free(blocked, goal)
    if goal is None:
        return None
    b = blocked.copy()
    b[max(start[0] - 2, 0):start[0] + 3, max(start[1] - 2, 0):start[1] + 3] = False   # we are standing here
    g = np.full((H, W), np.inf)
    parent = {}
    g[start] = 0.0
    heap = [(_octile(start, goal), 0.0, start)]
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if gc > g[cur]:
            continue
        for dy, dx, c in MOVES:
            y, x = cur[0] + dy, cur[1] + dx
            if not (0 <= y < H and 0 <= x < W) or b[y, x]:
                continue
            if dy and dx and (b[cur[0] + dy, cur[1]] or b[cur[0], cur[1] + dx]):
                continue                      # no squeezing between diagonal obstacles
            ng = gc + c
            if ng < g[y, x]:
                g[y, x] = ng
                parent[(y, x)] = cur
                heapq.heappush(heap, (ng + _octile((y, x), goal), ng, (y, x)))
    return None


def _line_free(blocked, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    for k in range(n + 1):
        t = k / n
        y, x = round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)
        if blocked[y, x]:
            return False
    return True


def shorten(blocked, path):
    """Drop waypoints that a straight, collision-free segment can skip."""
    if not path or len(path) < 3:
        return path
    b = blocked.copy()
    b[max(path[0][0] - 2, 0):path[0][0] + 3, max(path[0][1] - 2, 0):path[0][1] + 3] = False
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(b, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
