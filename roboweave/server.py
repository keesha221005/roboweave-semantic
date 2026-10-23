"""Simulation server for the dashboard.

    python -m roboweave.server --weights runs\\detect\\roboweave\\weights\\best.pt
    python -m roboweave.server                      # no weights: simulated detector

Endpoints
    GET  /state            latest snapshot (JSON)
    WS   /ws               same snapshot pushed ~5 times a second
    GET  /frame.jpg        robot camera with the tracker's boxes
    POST /control          {"action": "reset"|"pause"|"resume", "scenario", "mode", "vlm", "speed"}
    POST /compare          start the scenario x policy benchmark in the background
    GET  /compare          benchmark progress and results
    GET  /eval             summary from `python -m roboweave.evaluate` (if it has been run)
    GET  /health
The robot's decisions are made inside the Session; this process only steps it and reports.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from roboweave.scenarios import NAMES
from roboweave.session import DT, MODES, Session, run_headless

RESULTS_FILE = os.path.join("eval", "results.json")


class Control(BaseModel):
    action: str = "reset"
    scenario: str | None = None
    mode: str | None = None
    vlm: str | None = None
    speed: float | None = None


class Runner:
    """Owns the live Session and steps it on a background thread at (speed x) real time."""

    def __init__(self, weights=None, vlm="none", vlm_model=None, vlm_timeout=4.0):
        self.weights, self.vlm_model, self.vlm_timeout = weights, vlm_model, vlm_timeout
        self.cfg = {"scenario": "crossing", "mode": "roboweave", "vlm": vlm}
        self.speed, self.paused = 1.0, False
        self.lock = threading.Lock()
        self.session = self._new()
        self.error = None
        self.cmp = {"running": False, "done": 0, "total": 0, "results": [], "error": None}
        threading.Thread(target=self._loop, daemon=True).start()

    def _new(self):
        return Session(self.cfg["scenario"], self.cfg["mode"], self.weights, self.cfg["vlm"],
                       self.vlm_model, self.vlm_timeout)

    def _loop(self):
        nxt = time.monotonic()
        while True:
            now = time.monotonic()
            if self.paused or self.session.finished:
                time.sleep(0.05)
                nxt = time.monotonic()
                continue
            if now < nxt:
                time.sleep(min(nxt - now, 0.02))
                continue
            with self.lock:
                try:
                    self.session.step()
                except Exception as e:                       # keep the server alive, show the error
                    self.error = f"{type(e).__name__}: {e}"
                    self.paused = True
            nxt += DT / max(self.speed, 0.1)
            if nxt < time.monotonic() - 0.5:                 # fell behind (slow detector): do not burst
                nxt = time.monotonic()

    def control(self, c: Control):
        if c.action in ("pause", "resume"):
            self.paused = c.action == "pause"
            return
        if c.speed:
            self.speed = max(0.25, min(float(c.speed), 8.0))
        for k in ("scenario", "mode", "vlm"):
            v = getattr(c, k)
            if v:
                if (k == "scenario" and v not in NAMES) or (k == "mode" and v not in MODES) \
                        or (k == "vlm" and v not in ("none", "scripted", "ollama", "gemini")):
                    raise ValueError(f"bad {k}: {v}")
                self.cfg[k] = v
        if c.action == "reset":
            with self.lock:
                self.error, self.paused = None, False
                self.session = self._new()

    def snapshot(self):
        with self.lock:
            s = self.session.snapshot()
        s.update(paused=self.paused, speed=self.speed, error=self.error)
        return s

    def jpeg(self):
        with self.lock:
            return self.session.jpeg()

    # ----------------------------------------------------------- benchmark
    def start_compare(self):
        if self.cmp["running"]:
            return False
        jobs = [(sc, m) for sc in NAMES for m in MODES]
        self.cmp = {"running": True, "done": 0, "total": len(jobs), "results": [], "error": None}
        threading.Thread(target=self._compare, args=(jobs,), daemon=True).start()
        return True

    def _compare(self, jobs):
        try:
            for sc, m in jobs:
                vlm = self.cfg["vlm"] if m == "roboweave" else "none"
                self.cmp["results"].append(run_headless(sc, m, self.weights, vlm, self.vlm_model))
                self.cmp["done"] += 1
            os.makedirs("eval", exist_ok=True)
            with open(RESULTS_FILE, "w") as f:
                json.dump({"weights": self.weights, "vlm": self.cfg["vlm"], "results": self.cmp["results"]}, f)
        except Exception as e:
            self.cmp["error"] = f"{type(e).__name__}: {e}"
        self.cmp["running"] = False


def create_app(weights=None, vlm="none", vlm_model=None, vlm_timeout=4.0) -> FastAPI:
    app = FastAPI(title="RoboWeave sim server")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    run = Runner(weights, vlm, vlm_model, vlm_timeout)
    app.state.runner = run

    @app.get("/health")
    def health():
        return {"ok": True, "detector": "yolo" if weights else "simulated", "scenarios": list(NAMES), "modes": list(MODES), "config": run.cfg}

    @app.get("/state")
    def state():
        return run.snapshot()

    @app.get("/frame.jpg")
    def frame():
        return Response(run.jpeg(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/control")
    def control(c: Control):
        try:
            run.control(c)
        except ValueError as e:
            return Response(json.dumps({"error": str(e)}), status_code=400, media_type="application/json")
        return {"ok": True, "config": run.cfg, "paused": run.paused, "speed": run.speed}

    @app.post("/compare")
    def compare_start():
        return {"started": run.start_compare()}

    @app.get("/compare")
    def compare_status():
        saved = None
        if os.path.exists(RESULTS_FILE) and not run.cmp["results"]:
            try:
                saved = json.load(open(RESULTS_FILE))
            except (OSError, ValueError):
                saved = None
        return {**run.cmp, "saved": saved}

    @app.get("/eval")
    def eval_results():
        """Summary written by `python -m roboweave.evaluate` (without the raw runs)."""
        path = os.path.join("eval", "eval_results.json")
        if not os.path.exists(path):
            return {"available": False}
        try:
            d = json.load(open(path))
        except (OSError, ValueError):
            return {"available": False}
        return {"available": True, "meta": d["meta"], "summary": d["summary"]}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_text(json.dumps(run.snapshot()))
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--vlm", default="none", choices=["none", "scripted", "ollama", "gemini"])
    ap.add_argument("--vlm-model")
    ap.add_argument("--vlm-timeout", type=float, default=4.0)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(create_app(a.weights, a.vlm, a.vlm_model, a.vlm_timeout), host="127.0.0.1", port=a.port, log_level="warning")