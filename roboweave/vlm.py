"""Vision-language model advisor for scenes the detector cannot settle.

Rules this module is built around:
  * The safety loop NEVER waits on it. submit() returns at once; the answer shows up
    later through poll(). If it is slow, the robot just stays slow (see decision.py).
  * Rare calls: one request at a time, a cooldown between requests, one answer per scene.
  * One function behind four backends, swappable without touching the rest of the stack:
        ollama    local model on your machine. Free, offline, no data leaves the PC.  (recommended)
        gemini    Google Gemini API. Free tier is rate-limited and may be used to improve
                  Google models, so send simulated frames only.
        scripted  simulator oracle for tests (answers from the sim's ground truth).
        none      no VLM: decision.py stays cautious.
  * Standard library only (urllib + threading), no new dependencies.
"""
from __future__ import annotations

import base64
import io
import json
import os
import queue
import re
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass

PROMPT = (
    "You are the vision module of a warehouse robot. The image is from its front camera. "
    "Look at the scene ahead. Is there a person, or a static object such as a pillar, rack, box or pallet? "
    "Is an aisle blocked? Answer with ONLY this JSON and nothing else: "
    '{"object": "person|pillar|rack|box|pallet|other", "blocking_aisle": true|false, '
    '"advice": "stop|slow|route_around|proceed"}'
)

PERSON_WORDS = ("person", "human", "worker", "people", "pedestrian", "man", "woman")
STATIC_WORDS = ("pillar", "column", "pole", "rack", "shelf", "shelves", "box", "crate", "pallet",
                "cart", "wall", "equipment", "machine", "static")
ADVICE = ("stop", "slow", "route_around", "proceed")


@dataclass
class Advice:
    object: str = "unknown"          # person | static | unknown
    detail: str = ""                 # what the model called it
    blocking: bool = False
    advice: str = "slow"
    timeout: bool = False
    raw: str = ""

    @property
    def text(self) -> str:
        if self.timeout:
            return "no answer in time"
        return f"{self.detail or self.object}{', aisle blocked' if self.blocking else ''}, advice: {self.advice}"


def parse_advice(raw: str) -> Advice:
    """Tolerant parser: strict JSON if present, keyword fallback if the model rambled."""
    data = None
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
        except ValueError:
            data = None
    low = raw.lower()
    if isinstance(data, dict):
        obj = str(data.get("object", "")).lower()
        advice = str(data.get("advice", "")).lower().replace(" ", "_")
        blocking = data.get("blocking_aisle", False)
        blocking = blocking if isinstance(blocking, bool) else str(blocking).lower() == "true"
    else:
        obj, advice, blocking = low, "", ("block" in low)
    kind = ("person" if any(w in obj for w in PERSON_WORDS)
            else "static" if any(w in obj for w in STATIC_WORDS) else "unknown")
    if kind == "unknown" and not isinstance(data, dict):
        kind = "person" if any(w in low for w in PERSON_WORDS) else "static" if any(w in low for w in STATIC_WORDS) else "unknown"
    if advice not in ADVICE:
        advice = "stop" if kind == "person" else "slow"
    return Advice(kind, obj[:24], blocking, advice, False, raw[:200])


def encode_jpeg(frame, max_w=512) -> str:
    from PIL import Image
    im = Image.fromarray(frame)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- backends
class OllamaBackend:
    """Local vision model, e.g. `ollama pull gemma3:4b` (or any vision model you have pulled)."""
    name = "ollama"

    def __init__(self, model="gemma3:4b", url="http://localhost:11434"):
        self.model, self.url = model, url.rstrip("/")

    def ask(self, b64, prompt, timeout, hint=None) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "images": [b64],
                           "stream": False, "format": "json", "keep_alive": "30m",
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(self.url + "/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["response"]


class GeminiBackend:
    """Google Gemini REST API. Key is read from the GEMINI_API_KEY environment variable.
    Model names change over time: pass the current free-tier Flash model with --vlm-model."""
    name = "gemini"
    _warned = False

    def __init__(self, model="gemini-2.5-flash", key=None):
        self.model, self.key = model, key or os.environ.get("GEMINI_API_KEY", "")
        if not self.key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable to use --vlm gemini")
        if not GeminiBackend._warned:
            GeminiBackend._warned = True
            print("[vlm] Gemini free tier: Google may use submitted data to improve its models. "
                  "Only simulated frames are sent.")

    def ask(self, b64, prompt, timeout, hint=None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json", "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return out["candidates"][0]["content"]["parts"][0]["text"]


class ScriptedBackend:
    """Simulator oracle for logic tests: answers from the world's ground truth.
    `latency` is in SIMULATED seconds. Not a real VLM; never use it for results."""
    name = "scripted"

    def __init__(self, world, latency=1.0):
        self.world, self.latency = world, latency

    def ask(self, b64, prompt, timeout, hint=None) -> str:
        w, x, y = self.world, hint["x"], hint["y"]
        best, kind = 1.2, "other"
        for p in w.people:
            d = ((p.x - x) ** 2 + (p.y - y) ** 2) ** 0.5
            if d < best:
                best, kind = d, "person"
        for p in w.pillars:
            d = ((p.x - x) ** 2 + (p.y - y) ** 2) ** 0.5
            if d < best:
                best, kind = d, "pillar"
        for c in w.crates:
            d = ((min(max(x, c.x), c.x + c.w) - x) ** 2 + (min(max(y, c.y), c.y + c.h) - y) ** 2) ** 0.5
            if d < best:
                best, kind = d, "pallet"
        adv = "stop" if kind == "person" else "route_around"
        return json.dumps({"object": kind, "blocking_aisle": kind == "pallet", "advice": adv})


# ------------------------------------------------------------------- client
class VLMClient:
    """Non-blocking front end.

    threaded=True : real backends. A worker thread makes the call; poll() collects it.
    threaded=False: simulation mode. Time is the simulator's clock `t`, so tests are
                    deterministic and fast; backend.latency is in simulated seconds.
    """

    def __init__(self, backend, timeout=4.0, cooldown=2.0, threaded=True):
        self.backend, self.timeout, self.cooldown, self.threaded = backend, timeout, cooldown, threaded
        self._last = -1e9
        self._out: deque = deque()
        self._jobs: list = []                 # inline mode: (due, key, b64, hint)
        if threaded:
            self._q: queue.Queue = queue.Queue(maxsize=1)
            threading.Thread(target=self._worker, daemon=True).start()

    def _now(self, t):
        return time.monotonic() if self.threaded else t

    def submit(self, t, key, frame, hint=None) -> bool:
        now = self._now(t)
        if now - self._last < self.cooldown or self._busy():
            return False
        self._last = now
        b64 = encode_jpeg(frame)
        if self.threaded:
            try:
                self._q.put_nowait((key, b64, hint))
            except queue.Full:
                return False
        else:
            self._jobs.append((now + getattr(self.backend, "latency", 1.0), key, b64, hint))
        return True

    def _busy(self):
        return bool(self._jobs) if not self.threaded else self._q.full()

    def poll(self, t):
        if not self.threaded:
            now, still = t, []
            for due, key, b64, hint in self._jobs:
                if due - self._last > self.timeout:                      # would not answer in time
                    if now >= self._last + self.timeout:
                        self._out.append((key, Advice(timeout=True)))
                    else:
                        still.append((due, key, b64, hint))
                elif now >= due:
                    self._out.append((key, parse_advice(self.backend.ask(b64, PROMPT, self.timeout, hint))))
                else:
                    still.append((due, key, b64, hint))
            self._jobs = still
        out = list(self._out)
        self._out.clear()
        return out

    def _worker(self):
        while True:
            key, b64, hint = self._q.get()
            try:
                adv = parse_advice(self.backend.ask(b64, PROMPT, self.timeout, hint))
            except Exception as e:           # network down, model missing, timeout...
                adv = Advice(timeout=True, raw=f"{type(e).__name__}: {e}"[:200])
            self._out.append((key, adv))


def make_client(kind, model=None, world=None, latency=1.0, timeout=4.0):
    """Build the VLM front end from a CLI-style name; None means no VLM."""
    if kind in (None, "none"):
        return None
    if kind == "ollama":
        return VLMClient(OllamaBackend(model or "gemma3:4b"), timeout=timeout)
    if kind == "gemini":
        return VLMClient(GeminiBackend(model or "gemini-2.5-flash"), timeout=timeout)
    if kind == "scripted":
        return VLMClient(ScriptedBackend(world, latency), threaded=False)
    raise ValueError(f"unknown VLM backend {kind!r}")