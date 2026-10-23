"""Tiny HTTP client for the RoboWeave simulation server (python -m roboweave.server)."""
import os

import requests

BASE = os.environ.get("ROBOWEAVE_API", "http://127.0.0.1:8000").rstrip("/")


def get_json(path, timeout=1.5):
    try:
        r = requests.get(BASE + path, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def get_bytes(path, timeout=1.5):
    try:
        r = requests.get(BASE + path, timeout=timeout)
        r.raise_for_status()
        return r.content
    except requests.RequestException:
        return None


def post_json(path, payload=None, timeout=5):
    try:
        r = requests.post(BASE + path, json=payload or {}, timeout=timeout)
        return r.status_code, (r.json() if r.content else {})
    except (requests.RequestException, ValueError):
        return 0, {}