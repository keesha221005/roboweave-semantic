"""RoboWeave operator dashboard.

    # terminal 1 (project root)
    python -m roboweave.server --weights runs\\detect\\roboweave\\weights\\best.pt
    # terminal 2
    cd scada_dashboard
    streamlit run dashboard.py

The dashboard decides nothing: it only shows what the robot sees, decides and why.
"""
import pandas as pd
import streamlit as st

from components.rw_charts import (MODE_LABEL, compare_gap, compare_time, floor_map, gap_trace, speed_trace, state_donut)
from core import api

st.set_page_config(page_title="Roboweave | Operator console", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{ --bg:#07111b; --panel:#0d1b2a; --panel2:#10263a; --line:#1b3348; --text:#d6e4f0; --muted:#7f98ae;
       --cyan:#22d3ee; --green:#34d399; --amber:#f59e0b; --red:#ef4444; }
html, body, .stApp { background: radial-gradient(1200px 500px at 70% -10%, #0f2a3d 0%, var(--bg) 60%); color: var(--text); font-family:'IBM Plex Sans',sans-serif; }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:3.6rem; padding-bottom:1rem; max-width:1800px; }
section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] label p, section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{ color:var(--text) !important; }
section[data-testid="stSidebar"] h3{ font-family:'Space Grotesk',sans-serif; font-size:15px; color:#e6f1fa; }
div[data-testid="stVerticalBlockBorderWrapper"]{ background:linear-gradient(180deg,var(--panel) 0%, #0a1724 100%); border:1px solid var(--line); border-radius:12px; }
.panel-title{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:13px; color:#e6f1fa; margin:0 0 4px 0; line-height:1.3; }
.panel-sub{ font-size:11px; color:var(--muted); margin:-2px 0 6px 0; }
.topbar{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
.brand{ font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; letter-spacing:.3px; color:#f1f8ff; }
.brand small{ font-family:'IBM Plex Sans',sans-serif; font-weight:400; font-size:12px; color:var(--muted); margin-left:10px; }
.pill{ display:inline-block; padding:4px 11px; margin-left:6px; border-radius:999px; font-size:11px; font-weight:600; border:1px solid var(--line); background:var(--panel2); color:var(--text); }
.pill.ok{ color:var(--green); border-color:#1d5a49; background:#0c2a24; }
.pill.wait{ color:var(--amber); border-color:#5c4514; background:#2a2110; }
.pill.bad{ color:var(--red); border-color:#5c1d24; background:#341419; }
.kpi{ background:linear-gradient(180deg,var(--panel2) 0%, var(--panel) 100%); border:1px solid var(--line); border-radius:12px; padding:12px 16px; height:92px; box-sizing:border-box; }
.kpi .lbl{ font-size:12px; color:var(--muted); }
.kpi .val{ font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; color:#f1f8ff; line-height:1.15; margin-top:2px; }
.kpi .val span{ font-size:13px; font-weight:500; color:var(--muted); margin-left:4px; }
.kpi .sub{ font-size:11px; color:var(--muted); }
.kpi.accent{ border-color:#1e6a7d; box-shadow:0 0 0 1px rgba(34,211,238,.12) inset; }
.kpi.good .val{ color:var(--green); } .kpi.warn .val{ color:var(--amber); } .kpi.bad .val{ color:var(--red); }
.log{ font-family:'IBM Plex Sans',monospace; font-size:12px; color:var(--cyan); background:#08151f; border:1px solid var(--line); border-radius:6px; padding:4px 9px; margin-bottom:4px; }
.log b{ color:var(--muted); font-weight:500; margin-right:8px; }
.log.warn{ color:var(--amber); } .log.ok{ color:var(--green); } .log.bad{ color:var(--red); }
.obj{ display:flex; justify-content:space-between; font-size:12px; padding:6px 10px; margin-bottom:5px; border-radius:8px; background:var(--panel2); border-left:3px solid var(--muted); }
.obj.person{ border-left-color:var(--green); } .obj.pillar{ border-left-color:#60a5fa; } .obj.unsure{ border-left-style:dashed; }
.obj b{ font-weight:600; font-size:11px; color:var(--muted); }
[data-testid="stCaptionContainer"]{ color:var(--muted) !important; }
</style>
""", unsafe_allow_html=True)


def plot(fig, key):
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
    except TypeError:                                   # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def kpi(label, value, unit="", sub="", cls=""):
    return (f'<div class="kpi {cls}"><div class="lbl">{label}</div>'
            f'<div class="val">{value}<span>{unit}</span></div><div class="sub">{sub}</div></div>')


def log_class(msg):
    m = msg.lower()
    if "emergency" in m or "obstacle ahead" in m or "no clear path" in m:
        return "bad"
    if "yield" in m or "unclear" in m or "slow" in m or "caut" in m or "unavailable" in m or "error" in m:
        return "warn"
    if "clear" in m or "goal" in m or "routing" in m:
        return "ok"
    return ""


STATE_NOTE = {"GO": "driving", "YIELD": "yielding to a person", "STOP": "stopped for an obstacle",
              "WAIT": "no clear path", "DONE": "goal reached"}

# ---------------------------------------------------------------- sidebar
health = api.get_json("/health")
st.sidebar.markdown("### Mission console")
if not health:
    st.sidebar.error(f"No simulation server at {api.BASE}")
    st.sidebar.code("python -m roboweave.server", language="powershell")
else:
    scenarios, modes = health["scenarios"], health["modes"]
    cfg = health.get("config", {})
    vlms = ["none", "scripted", "ollama", "gemini"]
    scenario = st.sidebar.selectbox("Scenario", scenarios, index=scenarios.index(cfg.get("scenario", "crossing")),
                                    help="pillar: static pillar in the aisle. crossing: a person walks across. mixed: both. unclear: a crate YOLO was never trained on.")
    mode = st.sidebar.selectbox("Policy", modes, index=modes.index(cfg.get("mode", "roboweave")), format_func=lambda m: MODE_LABEL.get(m, m))
    vlm = st.sidebar.selectbox("VLM advisor", vlms, index=vlms.index(cfg.get("vlm", "none")),
                               help="Only used by RoboWeave. scripted = simulator oracle (logic test, not a real model). ollama = local model.")
    speed = st.sidebar.slider("Sim speed", 0.5, 4.0, 1.0, 0.5)
    b1, b2 = st.sidebar.columns(2)
    if b1.button("Start / reset", type="primary", width="stretch"):
        api.post_json("/control", {"action": "reset", "scenario": scenario, "mode": mode, "vlm": vlm, "speed": speed})
    if b2.button("Pause / resume", width="stretch"):
        cur = api.get_json("/state") or {}
        api.post_json("/control", {"action": "resume" if cur.get("paused") else "pause"})
    if st.sidebar.button("Apply speed", width="stretch"):
        api.post_json("/control", {"action": "speed", "speed": speed})
    st.sidebar.caption("Start / reset applies scenario, policy and VLM. The robot decides on its own; nothing here steers it.")

tab_live, tab_bench, tab_eval = st.tabs(["Live run", "Benchmark", "Evaluation"])


# ------------------------------------------------------------------- live
def live_view():
    s = api.get_json("/state")
    if not s:
        st.warning("Waiting for the simulation server.")
        return
    srv = '<span class="pill ok">sim server connected</span>'
    det = f'<span class="pill">{s["detector"]}</span>'
    vlm_pill = f'<span class="pill">VLM: {s["vlm"]}</span>'
    paused = '<span class="pill wait">paused</span>' if s["paused"] else ""
    st.markdown(f"""<div class="topbar"><div class="brand">Roboweave<small>Class-aware navigation &nbsp;/&nbsp; {s['scenario']} &nbsp;/&nbsp; {MODE_LABEL.get(s['mode'], s['mode'])}</small></div>
      <div>{srv}{det}{vlm_pill}{paused}</div></div>""", unsafe_allow_html=True)
    if s.get("error"):
        st.error(f"Simulation error: {s['error']}")

    gap = s["min_gap"]
    state = "FINISHED" if s["finished"] and s["done_t"] else ("TIMED OUT" if s["finished"] else s["state"])
    state_cls = {"YIELD": "warn", "STOP": "bad", "WAIT": "bad", "DONE": "good", "FINISHED": "good", "TIMED OUT": "bad"}.get(state, "accent")
    note = "reduced speed, scene unclear" if s["slow"] and s["state"] == "GO" else STATE_NOTE.get(s["state"], "")
    k = st.columns(5)
    k[0].markdown(kpi("Robot state", state.title(), "", note, state_cls), unsafe_allow_html=True)
    k[1].markdown(kpi("Run time", f"{s['t']:.1f}", "s", f"{s['speed']:.2f} m/s now"), unsafe_allow_html=True)
    k[2].markdown(kpi("Collisions", s["collisions"], "", "contact events with a person or object", "good" if s["collisions"] == 0 else "bad"), unsafe_allow_html=True)
    k[3].markdown(kpi("Closest to a person", "-" if gap is None else f"{gap:.2f}", "" if gap is None else "m", "smallest gap so far (surface to surface)"), unsafe_allow_html=True)
    k[4].markdown(kpi("To goal", f"{s['dist_to_goal']:.1f}", "m", "straight-line distance"), unsafe_allow_html=True)
    st.write("")

    col_map, col_side = st.columns([2.2, 1])
    with col_map:
        with st.container(border=True, height=470):
            st.markdown('<div class="panel-title">Warehouse floor (simulator view)</div>'
                        '<div class="panel-sub">Dotted amber ring = 2 m yield zone around each person. Green/blue rings = what the robot believes it sees (dashed = unsure).</div>', unsafe_allow_html=True)
            plot(floor_map(s), "floor")
        with st.container(border=True, height=316):
            st.markdown('<div class="panel-title">Decision log</div><div class="panel-sub">Every decision the robot makes, in plain English</div>', unsafe_allow_html=True)
            rows = s["log"][-8:][::-1]
            if rows:
                st.markdown("".join(f'<div class="log {log_class(m)}"><b>{t:5.1f}s</b>{m}</div>' for t, m in rows), unsafe_allow_html=True)
            else:
                st.caption("Nothing to explain yet: the robot has just been driving.")

    with col_side:
        with st.container(border=True, height=300):
            st.markdown('<div class="panel-title">Robot camera</div><div class="panel-sub">Tracker boxes: green = person, blue = static</div>', unsafe_allow_html=True)
            img = api.get_bytes("/frame.jpg")
            if img:
                st.image(img, width="stretch")
        with st.container(border=True, height=210):
            st.markdown('<div class="panel-title">What the robot believes</div>', unsafe_allow_html=True)
            if s["perceived"]:
                st.markdown("".join(
                    f'<div class="obj {o["cls"]} {"unsure" if o["unsure"] else ""}"><span>{o["cls"].title()}'
                    f'{" (unsure)" if o["unsure"] else ""}</span><b>{o["range"]:.1f} m &nbsp; conf {o["conf"]:.2f}</b></div>'
                    for o in sorted(s["perceived"], key=lambda o: o["range"])[:4]), unsafe_allow_html=True)
            else:
                st.caption("No objects currently tracked.")
        with st.container(border=True, height=260):
            st.markdown('<div class="panel-title">Time by state</div>', unsafe_allow_html=True)
            plot(state_donut(s["state_time"]), "donut")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True, height=260):
            st.markdown('<div class="panel-title">Speed</div><div class="panel-sub">Flat at zero = yielding or stopped</div>', unsafe_allow_html=True)
            plot(speed_trace(s["trace"]), "speed")
    with c2:
        with st.container(border=True, height=260):
            st.markdown('<div class="panel-title">Gap to nearest person</div><div class="panel-sub">Surface to surface, capped at 8 m</div>', unsafe_allow_html=True)
            plot(gap_trace(s["trace"]), "gap")


with tab_live:
    if hasattr(st, "fragment"):
        st.fragment(run_every=0.4)(live_view)()
    else:
        live_view()


# -------------------------------------------------------------- benchmark
def bench_view():
    c = api.get_json("/compare", timeout=3)
    if c is None:
        st.warning("Waiting for the simulation server.")
        return
    res = c["results"] or (c.get("saved") or {}).get("results") or []
    left, right = st.columns([1, 3])
    with left:
        if c["running"]:
            st.progress(c["done"] / max(c["total"], 1), text=f"Running {c['done']}/{c['total']} (headless, faster than real time)")
        elif st.button("Run benchmark", type="primary", width="stretch"):
            api.post_json("/compare")
            st.rerun()
        st.caption("Every scenario under all three policies, same world, same detector. Uses the server's detector and VLM setting.")
        if c.get("error"):
            st.error(c["error"])
    if not res:
        right.info("No results yet. Run the benchmark.")
        return
    t1, t2 = st.columns(2)
    with t1:
        with st.container(border=True, height=380):
            st.markdown('<div class="panel-title">Time to goal</div><div class="panel-sub">Bars at the top = did not finish in 60 s</div>', unsafe_allow_html=True)
            plot(compare_time(res), "cmp_t")
    with t2:
        with st.container(border=True, height=380):
            st.markdown('<div class="panel-title">Closest approach to a person</div><div class="panel-sub">Larger is safer. A policy that never finished (stop &amp; wait, mixed) never got near the person, so its bar is not a safety win</div>', unsafe_allow_html=True)
            plot(compare_gap(res), "cmp_g")
    rows = []
    for r in res:
        stopped = sum(v for k, v in r["state_time"].items() if k in ("STOP", "YIELD", "WAIT"))
        rows.append({"Scenario": r["scenario"], "Policy": MODE_LABEL[r["mode"]], "Finished": "yes" if r["time"] else "NO",
                     "Time (s)": f"{r['time']:.1f}" if r["time"] else ">60", "Collisions": r["collisions"],
                     "Closest to person (m)": "-" if r["min_gap"] is None else f"{r['min_gap']:.2f}", "Stopped (s)": round(stopped, 1),
                     "Slowed (s)": r["state_time"].get("slowed", 0.0)})
    with st.container(border=True):
        st.markdown('<div class="panel-title">Results table</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


with tab_bench:
    if hasattr(st, "fragment"):
        st.fragment(run_every=1.5)(bench_view)()
    else:
        bench_view()


# ------------------------------------------------------------- evaluation
LAB = {"roboweave": "RoboWeave", "box": "Lidar-only: route around", "stop_all": "Lidar-only: stop & wait"}


def _f(v, nd=None, unit=""):
    return "-" if v is None else (f"{v:.{nd}f}" if nd is not None else str(v)) + unit


with tab_eval:
    ev = api.get_json("/eval", timeout=3)
    if ev is None:
        st.warning("Waiting for the simulation server.")
    elif not ev.get("available"):
        st.info("No evaluation yet. In the project root run:")
        st.code("python -m roboweave.evaluate --weights runs\\detect\\roboweave\\weights\\best.pt", language="powershell")
    else:
        meta, summ = ev["meta"], ev["summary"]
        st.caption(f"{meta['seeds']} trials per cell. Detector: {meta['detector']}. Seed 0 is the nominal world; other seeds jitter lidar noise and the person's start delay, speed and direction.")
        tot = {m: [sum(summ[sc][m][k] for sc in summ) for k in ("collisions", "near_misses", "finished", "n")] for m in LAB}
        c = st.columns(3)
        for col, m in zip(c, LAB):
            col_, nm, fin, n = tot[m]
            col.markdown(kpi(LAB[m], f"{fin}/{n}", "runs finished", f"{col_} collisions, {nm} near misses",
                             "good" if (col_ == 0 and fin == n) else "warn"), unsafe_allow_html=True)
        st.write("")
        rows, rows2 = [], []
        for sc, by in summ.items():
            for m, s in by.items():
                t = "-" if s["time_mean"] is None else f"{s['time_mean']} ({s['time_min']}-{s['time_max']})"
                rows.append({"Scenario": sc, "Policy": LAB[m], "Finished": f"{s['finished']}/{s['n']}", "Time s (mean, range)": t,
                             "Collisions": s["collisions"], "Near misses": s["near_misses"],
                             "Closest to person (m)": _f(s["gap_min"], 2), "Moving in 2 m zone (s)": s["zone_moving_s"],
                             "Stopped s": s["stopped_mean"], "False yield s": s["false_yield_mean"], "Stuck s": s["stuck_mean"]})
                if m == "roboweave":
                    rows2.append({"Scenario": sc, "Person recall": _f(s["recall"], 1, "%"),
                                  "False person tracks (s)": s["false_person_s"], "Decisions logged": s["logged"]})
        with st.container(border=True):
            st.markdown('<div class="panel-title">Task and safety</div><div class="panel-sub">Negative closest approach = contact. Simulated people never avoid the robot</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.container(border=True):
            st.markdown('<div class="panel-title">Perception and explainability (RoboWeave)</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows2), hide_index=True, width="stretch")