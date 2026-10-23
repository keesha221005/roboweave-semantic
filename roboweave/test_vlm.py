"""Checks the VLM client against a tiny fake Ollama server. No Ollama or network needed.

    python -m roboweave.test_vlm
"""
import http.server
import json
import threading
import time

import numpy as np

from roboweave.vlm import OllamaBackend, VLMClient, parse_advice

MODE = {"v": "ok"}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n))
        assert body["images"] and body["format"] == "json" and body["stream"] is False   # request shape
        if MODE["v"] == "slow": time.sleep(3)
        txt = {"ok": '{"object": "person", "blocking_aisle": false, "advice": "stop"}',
               "slow": '{"object": "box", "blocking_aisle": true, "advice": "route_around"}',
               "ramble": "I can see a worker in a yellow vest standing in the aisle, you should stop."}[MODE["v"]]
        out = json.dumps({"response": txt}).encode()
        try:
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
        except BrokenPipeError:      # the client gave up (that is the timeout case we are testing)
            pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 18434), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
frame = np.zeros((360, 640, 3), np.uint8)

def run(name, backend_url, mode, timeout, wait):
    MODE["v"] = mode
    c = VLMClient(OllamaBackend("test", backend_url), timeout=timeout, cooldown=0.0)
    t0 = time.monotonic(); ok = c.submit(0, ("k", 1), frame, {})
    submit_ms = (time.monotonic() - t0) * 1000
    time.sleep(wait); res = c.poll(0)
    print(f"{name:28s} submit returned in {submit_ms:5.1f} ms  accepted={ok}  ->",
          [(k, a.object, a.advice, "TIMEOUT" if a.timeout else "") for k, a in res] or "no answer yet")

run("good answer", "http://127.0.0.1:18434", "ok", 4, 1.0)
run("rambling (no JSON)", "http://127.0.0.1:18434", "ramble", 4, 1.0)
run("server slower than timeout", "http://127.0.0.1:18434", "slow", 1.0, 2.0)
run("server down", "http://127.0.0.1:1", "ok", 2, 1.0)
print()
for raw in ['{"object":"Pillar","blocking_aisle":"false","advice":"route around"}',
            '```json\n{"object": "person", "advice": "stop"}\n```', "a forklift", "", "not json {broken"]:
    a = parse_advice(raw); print(f"{raw[:48]!r:52} -> {a.object:8s} advice={a.advice} blocking={a.blocking}")
