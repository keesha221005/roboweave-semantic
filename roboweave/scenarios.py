"""Scripted scenarios: same map and same start/goal for every policy."""
from __future__ import annotations

from roboweave.world import Crate, Person, Pillar, Robot, default_world


def build(name: str, seed=None):
    """Return (world, goal). Robot starts at the west end of the main aisle.
    With a seed, walking people get a random start delay, speed and direction (for repeated trials)."""
    w = default_world()
    w.robot = Robot(1.5, 7.0, 0.0)
    goal = (22.5, 7.0)
    if name == "pillar":                       # a pillar in the aisle, no people
        w.people = []
        w.pillars = [Pillar(11.2, 6.4)]
    elif name == "crossing":                   # a person steps across the aisle as the robot arrives
        w.pillars = []
        p = Person(12.0, 13.0, route=[(12.0, 13.0), (12.0, 1.0)], speed=0.8, pause=0.0)
        p._wait = 1.0
        w.people = [p]
    elif name == "mixed":                      # pillar plus a crossing person
        w.pillars = [Pillar(8.0, 6.5)]
        p = Person(12.0, 13.0, route=[(12.0, 13.0), (12.0, 1.0)], speed=0.8, pause=0.0)
        p._wait = 1.0
        w.people = [p]
    elif name == "unclear":                    # a pallet seals the east end of the main aisle
        w.people, w.pillars = [], []
        w.crates = [Crate(16.0, 5.9, 1.2, 2.5)]
    else:
        raise ValueError(f"unknown scenario {name!r}")
    if seed is not None:
        import random
        rng = random.Random(seed * 7 + 1)
        for p in w.people:
            p._wait = rng.uniform(0.0, 3.0)
            p.speed = rng.uniform(0.7, 0.9)
            if rng.random() < 0.5:
                p.route = p.route[::-1]
                p.x, p.y = p.route[0]
    w._build_segments()
    return w, goal


NAMES = ["pillar", "crossing", "mixed", "unclear"]