"""Manual drive demo: top-down map + lidar rays + camera view.

Run:   python -m roboweave.demo
Keys:  arrows / WASD drive, SPACE pause people, ESC quit.
Test:  python -m roboweave.demo --headless --frames 120   (saves camera_test.png)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--frames", type=int, default=0)
args = ap.parse_args()
if args.headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from roboweave.sensors import Camera, Lidar  # noqa: E402
from roboweave.world import default_world  # noqa: E402

SCALE = 40            # px per metre
DT = 1 / 30


def to_px(x, y, h):
    return int(x * SCALE), int((h - y) * SCALE)   # flip y so +y is up on screen


def main():
    pygame.init()
    world, lidar, cam = default_world(), Lidar(), Camera()
    mw, mh = int(world.width * SCALE), int(world.height * SCALE)
    cw = 480
    screen = pygame.display.set_mode((mw + cw, mh))
    pygame.display.set_caption("RoboWeave sim")
    font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    paused, frame_i = False, 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
        k = pygame.key.get_pressed()
        v = (k[pygame.K_UP] or k[pygame.K_w]) - (k[pygame.K_DOWN] or k[pygame.K_s])
        w = (k[pygame.K_LEFT] or k[pygame.K_a]) - (k[pygame.K_RIGHT] or k[pygame.K_d])
        if args.headless:                      # scripted: drive straight
            v, w = 1, 0
        if paused:
            saved = [(p, p.speed) for p in world.people]
            for p, _ in saved:
                p.speed = 0
        world.step(DT, v * world.robot.max_v, w * world.robot.max_w)
        if paused:
            for p, sp in saved:
                p.speed = sp

        ranges = lidar.scan(world)
        frame, gt = cam.render(world)

        screen.fill((18, 22, 30))
        rb = world.robot
        for rk in world.racks:
            x, y = to_px(rk.x, rk.y + rk.h, world.height)
            pygame.draw.rect(screen, (196, 120, 50), (x, y, rk.w * SCALE, rk.h * SCALE))
        for p in world.pillars:
            pygame.draw.circle(screen, (150, 150, 155), to_px(p.x, p.y, world.height), int(p.r * SCALE))
        for p in world.people:
            pygame.draw.circle(screen, (240, 190, 20), to_px(p.x, p.y, world.height), int(p.r * SCALE))
        o = to_px(rb.x, rb.y, world.height)
        for a, d in zip(lidar.rel_angles[::3], ranges[::3]):
            dd = d if np.isfinite(d) else lidar.max_range
            end = to_px(rb.x + dd * math.cos(rb.theta + a), rb.y + dd * math.sin(rb.theta + a), world.height)
            pygame.draw.line(screen, (30, 90, 110), o, end, 1)
        pygame.draw.circle(screen, (34, 211, 238), o, int(rb.r * SCALE))
        nose = to_px(rb.x + rb.r * math.cos(rb.theta), rb.y + rb.r * math.sin(rb.theta), world.height)
        pygame.draw.line(screen, (255, 255, 255), o, nose, 3)

        img = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        for g in gt:                           # ground-truth boxes (debug only)
            x1, y1, x2, y2 = g["bbox"]
            pygame.draw.rect(img, (0, 255, 0) if g["cls"] == "person" else (80, 160, 255),
                             (x1, y1, x2 - x1, y2 - y1), 2)
        img = pygame.transform.smoothscale(img, (cw, int(cw * cam.H / cam.W)))
        screen.blit(img, (mw, 0))
        info = [f"t={world.time:5.1f}s  collisions={world.collisions}",
                f"min lidar range={np.nanmin(ranges):.2f} m",
                "camera GT boxes: green=person blue=pillar"]
        for i, s in enumerate(info):
            screen.blit(font.render(s, True, (214, 228, 240)), (mw + 8, img.get_height() + 10 + i * 18))
        pygame.display.flip()
        clock.tick(30 if not args.headless else 0)

        frame_i += 1
        if args.frames and frame_i >= args.frames:
            pygame.image.save(screen, "screen_test.png")
            pygame.image.save(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), "camera_test.png")
            print("frames", frame_i, "collisions", world.collisions, "gt", [g["cls"] for g in gt])
            return


if __name__ == "__main__":
    main()
