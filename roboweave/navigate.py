"""Closed-loop navigation: sensors -> perception -> grid -> decision -> motion.

    python -m roboweave.navigate --scenario crossing --mode roboweave
    python -m roboweave.navigate --scenario crossing --mode stop_all --headless
    python -m roboweave.navigate --scenario mixed --weights runs\\detect\\roboweave\\weights\\best.pt

Without --weights a noisy fake detector (ground-truth boxes with drops and label
flips) stands in for YOLO, which is fine for logic checks but not a real result.
"""
from __future__ import annotations

import argparse
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="crossing")
ap.add_argument("--mode", default="roboweave", choices=["roboweave", "stop_all", "box"])
ap.add_argument("--weights")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--timeout", type=float, default=90.0, help="sim seconds")
ap.add_argument("--perfect", action="store_true", help="fake detector without noise")
ap.add_argument("--vlm", default="none", choices=["none", "scripted", "ollama", "gemini"])
ap.add_argument("--vlm-model", help="e.g. gemma3:4b for ollama; a current Flash model for gemini")
ap.add_argument("--vlm-timeout", type=float, default=4.0, help="seconds to wait for a real VLM before giving up")
ap.add_argument("--vlm-latency", type=float, default=1.0, help="scripted VLM answer delay, simulated seconds")
args = ap.parse_args()
if args.headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np  # noqa: E402

from roboweave.decision import Navigator  # noqa: E402
from roboweave.grid import PERSON, PILLAR, UNKNOWN  # noqa: E402
from roboweave.perception import NoisyGTDetector, Perception, YoloDetector  # noqa: E402
from roboweave.scenarios import build  # noqa: E402
from roboweave.sensors import Camera, Lidar  # noqa: E402
from roboweave.vlm import make_client  # noqa: E402

DT = 1 / 30
SCALE = 40


def run(scenario, mode, weights=None, headless=True, timeout=90.0, perfect=False, quiet=False,
        vlm="none", vlm_model=None, vlm_latency=1.0, vlm_timeout=4.0):
    import pygame
    world, goal = build(scenario)
    lidar, cam = Lidar(seed=3), Camera()
    fake = None if weights else (NoisyGTDetector(drop=0, flip=0, false_pos=0) if perfect else NoisyGTDetector())
    perc = Perception(YoloDetector(weights) if weights else fake, cam, lidar)
    nav = Navigator(world, mode, lidar, make_client(vlm, vlm_model, world, vlm_latency, vlm_timeout))

    screen = font = None
    if not headless:
        pygame.init()
        mw, mh = int(world.width * SCALE), int(world.height * SCALE)
        screen = pygame.display.set_mode((mw + 480, mh))
        pygame.display.set_caption(f"RoboWeave nav: {scenario} / {mode}")
        font = pygame.font.SysFont("consolas", 14)
        clock = pygame.time.Clock()

    t, done_t, stuck_since = 0.0, None, None
    state_time = {}
    while t < timeout:
        if not headless:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return None
        ranges = lidar.scan(world)
        frame, gt = cam.render(world)
        if fake:
            fake.gt = gt
        obs = perc.update(frame, ranges, world.robot, t)
        cmd = nav.step(t, world.robot, ranges, obs, goal, frame)
        state_time[cmd.state] = state_time.get(cmd.state, 0.0) + DT
        if cmd.state == "GO" and cmd.slow < 1.0:
            state_time["slowed"] = state_time.get("slowed", 0.0) + DT
        world.step(DT, cmd.v, cmd.w)
        t += DT
        if cmd.state == "DONE":
            done_t = t
            break
        if not headless:
            _draw(screen, font, world, nav, obs, frame, perc, t, cmd, goal, cam)
            clock.tick(30)

    res = {"scenario": scenario, "mode": mode, "time": done_t, "collisions": world.collisions,
           "min_gap": world.min_person_gap, "state_time": state_time, "log": nav.log}
    if not quiet:
        print(f"\n{scenario} / {mode}: " + (f"reached goal in {done_t:.1f} s" if done_t else f"DID NOT FINISH in {timeout:.0f} s"))
        print(f"collisions {world.collisions}   closest approach to a person "
              + ("-" if math.isinf(world.min_person_gap) else f"{world.min_person_gap:.2f} m"))
        print("time in state:", {k: round(v, 1) for k, v in state_time.items()}, "  (slowed = part of GO at reduced speed)")
        print("decision log:")
        for tt, msg in nav.log:
            print(f"  {tt:5.1f}s  {msg}")
    return res


def _draw(screen, font, world, nav, obs, frame, perc, t, cmd, goal, cam):
    import pygame
    H = world.height
    px = lambda x, y: (int(x * SCALE), int((H - y) * SCALE))
    screen.fill((18, 22, 30))
    gf = nav.frame
    if gf is not None:
        res = nav.grid.res
        colors = {PERSON: (90, 60, 20), PILLAR: (40, 55, 90), UNKNOWN: (70, 40, 70)}
        for lab, c in colors.items():
            for j, i in zip(*np.nonzero(gf.semantic == lab)):
                pygame.draw.rect(screen, c, (int(i * res * SCALE), int((H - (j + 1) * res) * SCALE),
                                             int(res * SCALE) + 1, int(res * SCALE) + 1))
    for rk in world.racks:
        x, y = px(rk.x, rk.y + rk.h)
        pygame.draw.rect(screen, (196, 120, 50), (x, y, rk.w * SCALE, rk.h * SCALE))
    for p in world.pillars:
        pygame.draw.circle(screen, (150, 150, 155), px(p.x, p.y), int(p.r * SCALE))
    for p in world.people:
        pygame.draw.circle(screen, (240, 190, 20), px(p.x, p.y), int(p.r * SCALE))
    for o in obs:
        pygame.draw.circle(screen, (0, 255, 0) if o.cls == "person" else (80, 160, 255), px(o.x, o.y),
                           int(0.45 * SCALE), 2)
    if nav.path:
        pts = [px(world.robot.x, world.robot.y)] + [px(x, y) for x, y in nav.path]
        pygame.draw.lines(screen, (34, 211, 238), False, pts, 2)
    pygame.draw.circle(screen, (255, 80, 80), px(*goal), 8, 2)
    rb = world.robot
    pygame.draw.circle(screen, (34, 211, 238), px(rb.x, rb.y), int(rb.r * SCALE))
    pygame.draw.line(screen, (255, 255, 255), px(rb.x, rb.y),
                     px(rb.x + rb.r * math.cos(rb.theta), rb.y + rb.r * math.sin(rb.theta)), 3)

    mw = int(world.width * SCALE)
    img = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    for tr in perc.tracker.tracks:
        if tr.hits >= perc.tracker.min_hits:
            x1, y1, x2, y2 = tr.bbox
            col = (0, 255, 0) if tr.cls == "person" else (80, 160, 255)
            pygame.draw.rect(img, col, (x1, y1, x2 - x1, y2 - y1), 2)
            img.blit(font.render(f"{tr.cls} {tr.conf:.2f}", True, col), (x1, max(0, y1 - 16)))
    img = pygame.transform.smoothscale(img, (480, int(480 * cam.H / cam.W)))
    screen.blit(img, (mw, 0))
    y0 = img.get_height() + 8
    head = f"{nav.mode.upper()}  t={t:5.1f}s  state={cmd.state}  v={cmd.v:.2f}"
    screen.blit(font.render(head, True, (214, 228, 240)), (mw + 8, y0))
    screen.blit(font.render(f"collisions={world.collisions}", True, (214, 228, 240)), (mw + 8, y0 + 18))
    screen.blit(font.render("DECISION LOG", True, (127, 152, 174)), (mw + 8, y0 + 46))
    for k, (tt, msg) in enumerate(nav.log[-9:]):
        screen.blit(font.render(f"{tt:5.1f}s {msg}"[:62], True, (34, 211, 238)), (mw + 8, y0 + 66 + k * 18))
    pygame.display.flip()


if __name__ == "__main__":
    run(args.scenario, args.mode, args.weights, args.headless, args.timeout, args.perfect,
        vlm=args.vlm, vlm_model=args.vlm_model, vlm_latency=args.vlm_latency, vlm_timeout=args.vlm_timeout)