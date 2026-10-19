import time
from collections import OrderedDict

import pandas as pd
import streamlit as st

from core.cad_parser import DEFAULT_GRID, parse_cad_image
from core.bridge import init_ros_engine
from components.ui_charts import (
    render_floorplan_chart,
    render_telemetry_breakdown_chart,
    render_battery_chart,
    render_service_times_chart,
    render_output_chart,
    render_health_gauge,
)

st.set_page_config(page_title="Roboweave | Fleet Control", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{
  --bg:#07111b; --panel:#0d1b2a; --panel2:#10263a; --line:#1b3348;
  --text:#d6e4f0; --muted:#7f98ae; --cyan:#22d3ee; --teal:#2dd4bf;
  --green:#34d399; --amber:#f59e0b; --red:#ef4444;
}
html, body, .stApp { background: radial-gradient(1200px 500px at 70% -10%, #0f2a3d 0%, var(--bg) 60%); color: var(--text); font-family:'IBM Plex Sans',sans-serif; }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:1.4rem; padding-bottom:1rem; max-width:1800px; }
section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] h3{ font-family:'Space Grotesk',sans-serif; font-size:15px; color:#e6f1fa; }

/* bordered containers become panels */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(180deg,var(--panel) 0%, #0a1724 100%);
  border:1px solid var(--line); border-radius:12px;
}
.panel-title{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:13px; color:#e6f1fa; margin:0 0 4px 0; line-height:1.3; }
.panel-sub{ font-size:11px; color:var(--muted); margin:-2px 0 6px 0; }

/* top bar */
.topbar{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
.brand{ font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; letter-spacing:.3px; color:#f1f8ff; }
.brand small{ font-family:'IBM Plex Sans',sans-serif; font-weight:400; font-size:12px; color:var(--muted); margin-left:10px; }
.pill{ display:inline-block; padding:4px 11px; margin-left:6px; border-radius:999px; font-size:11px; font-weight:600;
       border:1px solid var(--line); background:var(--panel2); color:var(--text); }
.pill.ok{ color:var(--green); border-color:#1d5a49; background:#0c2a24; }
.pill.wait{ color:var(--amber); border-color:#5c4514; background:#2a2110; }

/* KPI cards */
.kpi{ background:linear-gradient(180deg,var(--panel2) 0%, var(--panel) 100%); border:1px solid var(--line);
      border-radius:12px; padding:12px 16px; height:92px; box-sizing:border-box; }
.kpi .lbl{ font-size:12px; color:var(--muted); }
.kpi .val{ font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; color:#f1f8ff; line-height:1.15; margin-top:2px; }
.kpi .val span{ font-size:13px; font-weight:500; color:var(--muted); margin-left:4px; }
.kpi .sub{ font-size:11px; color:var(--muted); }
.kpi.accent{ border-color:#1e6a7d; box-shadow:0 0 0 1px rgba(34,211,238,.12) inset; }

/* fleet cards */
.amr{ border:1px solid var(--line); background:var(--panel2); border-radius:10px; padding:9px 12px; margin-bottom:8px; }
.amr .row{ display:flex; justify-content:space-between; align-items:center; }
.amr .name{ font-family:'Space Grotesk',sans-serif; font-weight:600; color:#f1f8ff; }
.amr .item{ font-size:11px; color:var(--muted); margin:2px 0 6px; }
.bar{ height:6px; border-radius:4px; background:#0a1724; overflow:hidden; }
.bar > div{ height:100%; border-radius:4px; }
.tag{ font-size:10px; font-weight:600; padding:2px 8px; border-radius:999px; }
.tag.NAVIGATING{ color:var(--cyan); background:#0b2d38; }
.tag.COMPLETED,.tag.STANDBY{ color:var(--green); background:#0c2a24; }
.tag.BLOCKED{ color:var(--red); background:#341419; }

/* alerts */
.alert{ display:flex; justify-content:space-between; font-size:12px; padding:7px 10px; margin-bottom:6px;
        border-radius:8px; background:var(--panel2); border-left:3px solid var(--muted); }
.alert.high{ border-left-color:var(--red); } .alert.med{ border-left-color:var(--amber); }
.alert.info{ border-left-color:var(--cyan); } .alert.ok{ border-left-color:var(--green); }
.alert b{ font-weight:600; font-size:11px; }

.mesh{ font-family:'IBM Plex Sans',monospace; font-size:11.5px; color:var(--cyan); background:#08151f;
       border:1px solid var(--line); border-radius:6px; padding:4px 8px; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bridge():
    return init_ros_engine()


ros_node, ros_data = get_bridge()

if "grid" not in st.session_state:
    st.session_state.grid = DEFAULT_GRID
if "cad_name" not in st.session_state:
    st.session_state.cad_name = "DEFAULT_WAREHOUSE.dwg"

# ------------------------------------------------- sidebar (logic unchanged)
st.sidebar.markdown("### Mission console")

with ros_data["lock"]:
    available_robots = list(ros_data["fleet"].keys())

if available_robots:
    selected_robots = st.sidebar.multiselect(
        "Select target AMRs", available_robots,
        default=available_robots[:1] if available_robots else None)

    active_grid = st.session_state.grid
    gh, gw = len(active_grid), len(active_grid[0])

    dynamic_stations = {
        "INBOUND_DOCK (Row 0, Col 0)": [0, 0],
        f"OUTBOUND_BAY (Row {gh-1}, Col {gw-1})": [gh - 1, gw - 1],
        f"CENTRAL_CORRIDOR (Row {gh//2}, Col {gw//2})": [gh // 2, gw // 2],
        f"NORTH_FEEDER (Row 0, Col {gw-1})": [0, gw - 1],
        f"SOUTH_BUFFER (Row {gh-1}, Col 0)": [gh - 1, 0],
    }

    dispatch_mode = st.sidebar.radio("Destination mode",
                                     ["Dynamic Bay Presets", "Custom Floor Coordinates"], horizontal=True)
    dest_coords = [0, 0]
    is_valid_point = True

    if dispatch_mode == "Dynamic Bay Presets":
        selected_stn = st.sidebar.selectbox("Destination action point", list(dynamic_stations.keys()))
        dest_coords = dynamic_stations[selected_stn]
    else:
        c1, c2 = st.sidebar.columns(2)
        target_r = c1.number_input("Target row", 0, gh - 1, gh // 2)
        target_c = c2.number_input("Target col", 0, gw - 1, gw // 2)
        dest_coords = [int(target_r), int(target_c)]
        if active_grid[dest_coords[0]][dest_coords[1]] == 1:
            st.sidebar.error("Selected coordinate is inside a structural obstacle!")
            is_valid_point = False

    sku_name = st.sidebar.text_input("Payload SKU / cargo", "Medical Supplies (SKU-89)")
    prio = st.sidebar.slider("SLA dispatch priority", min_value=1, max_value=3, value=2)

    can_dispatch = is_valid_point and len(selected_robots) > 0
    if st.sidebar.button("Dispatch mission over DDS", type="primary",
                         use_container_width=True, disabled=not can_dispatch):
        for robot in selected_robots:
            ros_node.dispatch_mission(robot, dest_coords, prio, sku_name)
        st.sidebar.success(f"Dispatched {', '.join(selected_robots)} to {dest_coords}")
else:
    st.sidebar.warning("Launch ROS 2 nodes to enable dynamic dispatching.")

st.sidebar.divider()
st.sidebar.markdown("### CAD floorplan overlay")
uploaded_cad = st.sidebar.file_uploader("Upload blueprint", type=["png", "jpg", "jpeg", "bmp"])
if uploaded_cad and st.sidebar.button("Compile CAD"):
    st.session_state.grid = parse_cad_image(uploaded_cad)
    st.session_state.cad_name = uploaded_cad.name
    st.sidebar.success("Floorplan re-compiled!")
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### Tactical disruptions")
with st.sidebar.form("obstacle_form"):
    current_gh, current_gw = len(st.session_state.grid), len(st.session_state.grid[0])
    obs_r = st.number_input("Grid row (Y)", min_value=0, max_value=current_gh - 1, value=min(4, current_gh - 1))
    obs_c = st.number_input("Grid col (X)", min_value=0, max_value=current_gw - 1, value=min(7, current_gw - 1))
    col_inj, col_clr = st.columns(2)
    with col_inj:
        inject = st.form_submit_button("Inject blockage")
    with col_clr:
        clear = st.form_submit_button("Clear all")

if inject:
    ros_node.inject_obstacle(int(obs_r), int(obs_c))
    st.sidebar.success(f"Obstacle locked at [{int(obs_r)}, {int(obs_c)}]")
if clear:
    ros_node.clear_obstacles()
    st.sidebar.info("Mesh cleared.")

st.sidebar.divider()
st.sidebar.markdown("### WMS task queue")
with st.sidebar.form("queue_form"):
    active_grid = st.session_state.grid
    q_gh, q_gw = len(active_grid), len(active_grid[0])
    q_r = st.number_input("Queue target row", 0, q_gh - 1, 0)
    q_c = st.number_input("Queue target col", 0, q_gw - 1, q_gw - 1)
    q_sku = st.text_input("Queue item SKU", "Automated Cargo (SKU-99)")
    q_prio = st.slider("Queue priority", 1, 3, 2)
    add_queue = st.form_submit_button("Enqueue task")

if add_queue:
    ros_node.add_task_to_queue([int(q_r), int(q_c)], int(q_prio), q_sku)
    st.sidebar.success(f"Task enqueued for target [{q_r}, {q_c}]")


# ----------------------------------------------------------------- helpers
def kpi(label, value, unit="", sub="", accent=False):
    cls = "kpi accent" if accent else "kpi"
    return (f'<div class="{cls}"><div class="lbl">{label}</div>'
            f'<div class="val">{value}<span>{unit}</span></div><div class="sub">{sub}</div></div>')


def deliveries_per_minute(history, limit=15):
    counts = OrderedDict()
    for row in reversed(history):              # history is newest-first
        key = str(row.get("Timestamp", ""))[:5]  # HH:MM
        counts[key] = counts.get(key, 0) + 1
    return list(counts.items())[-limit:]


def build_alerts(fleet, n_obstacles):
    alerts = []
    for r_id, amr in fleet.items():
        if amr["status"] == "BLOCKED":
            alerts.append(("high", f"{r_id} is blocked", "High"))
        if amr["battery"] < 20:
            alerts.append(("med", f"{r_id} battery at {amr['battery']:.0f}%", "Medium"))
    if n_obstacles:
        alerts.append(("info", f"{n_obstacles} injected blockage(s) active", "Info"))
    if not alerts:
        alerts.append(("ok", "All AMRs operating normally", "Clear"))
    return alerts


# --------------------------------------------------------------- live view
def live_view():
    with ros_data["lock"]:
        fleet = {k: v.copy() for k, v in ros_data["fleet"].items()}
        history = list(ros_data["transport_history"])
        logs = list(ros_data["mesh_logs"])
        stats = dict(ros_data["action_point_stats"])
        queue_depth = len(ros_node.task_queue)
        n_obstacles = len(ros_node.active_obstacles)

    robots = list(fleet.keys())
    n = len(robots)
    n_blocked = sum(1 for a in fleet.values() if a["status"] == "BLOCKED")
    avg_batt = sum(a["battery"] for a in fleet.values()) / n if n else 0
    orders = len(history)
    availability = round((n - n_blocked) / n * 100) if n else 0

    # top bar
    live_pill = '<span class="pill ok">ROS 2 DDS active</span>' if n else '<span class="pill wait">Awaiting nodes</span>'
    st.markdown(f"""
    <div class="topbar">
      <div class="brand">Roboweave<small>Fleet control &nbsp;/&nbsp; {st.session_state.cad_name}</small></div>
      <div>{live_pill}<span class="pill">{n} AMRs connected</span><span class="pill ok">WMS dispatch enabled</span></div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    k = st.columns(4)
    k[0].markdown(kpi("Connected AMRs", f"{n}", "", "reporting over ROS 2", True), unsafe_allow_html=True)
    k[1].markdown(kpi("Fleet charge", f"{avg_batt:.0f}", "%", "average state of charge"), unsafe_allow_html=True)
    k[2].markdown(kpi("Orders done", f"{orders}", "", f"{queue_depth} waiting in queue"), unsafe_allow_html=True)
    k[3].markdown(kpi("Blocked AMRs", f"{n_blocked}", f"/ {n}", f"{n_obstacles} injected blockages"), unsafe_allow_html=True)

    st.write("")
    col_map, col_side = st.columns([2.2, 1])

    # Heights are fixed so the left and right stacks end on the same line:
    # 490 + 150 + gap  ==  250 + 150 + 224 + 2 gaps  (gap = 16px)
    with col_map:
        with st.container(border=True, height=490):
            st.markdown('<div class="panel-title">Live floor map</div>'
                        '<div class="panel-sub">Real-time AMR positions, planned paths and destinations</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(render_floorplan_chart(st.session_state.grid, fleet),
                            use_container_width=True, key="floor")
        with st.container(border=True, height=150):
            st.markdown('<div class="panel-title">Peer-to-peer mesh</div>', unsafe_allow_html=True)
            if logs:
                st.markdown("".join(f'<div class="mesh">{l}</div>' for l in logs[-3:]), unsafe_allow_html=True)
            else:
                st.caption("Awaiting reservation broadcasts from active nodes...")

    with col_side:
        with st.container(border=True, height=250):
            st.markdown('<div class="panel-title">Fleet status</div>', unsafe_allow_html=True)
            if not fleet:
                st.caption("No AMRs connected yet.")
            for r_id, a in fleet.items():
                b = a["battery"]
                bc = "#ef4444" if b < 20 else "#f59e0b" if b < 50 else "#34d399"
                st.markdown(f"""
                <div class="amr">
                  <div class="row"><span class="name">{r_id}</span><span class="tag {a['status']}">{a['status'].title()}</span></div>
                  <div class="item">{a.get('item', 'Cargo')} &nbsp;→&nbsp; {list(a['goal'])}</div>
                  <div class="bar"><div style="width:{b:.0f}%; background:{bc};"></div></div>
                </div>""", unsafe_allow_html=True)
        with st.container(border=True, height=150):
            st.markdown('<div class="panel-title">Alerts</div>', unsafe_allow_html=True)
            st.markdown("".join(f'<div class="alert {c}"><span>{t}</span><b>{s}</b></div>'
                                for c, t, s in build_alerts(fleet, n_obstacles)), unsafe_allow_html=True)
        with st.container(border=True, height=224):
            st.markdown('<div class="panel-title">Fleet availability</div>'
                        '<div class="panel-sub">Share of AMRs not blocked</div>', unsafe_allow_html=True)
            st.plotly_chart(render_health_gauge(availability), use_container_width=True, key="gauge")

    # Bottom row: four equal panels, identical height and chart margins
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        with st.container(border=True, height=250):
            st.markdown('<div class="panel-title">Deliveries per minute</div>', unsafe_allow_html=True)
            st.plotly_chart(render_output_chart(deliveries_per_minute(history)),
                            use_container_width=True, key="output")
    with b2:
        with st.container(border=True, height=250):
            st.markdown('<div class="panel-title">Operational breakdown (%)</div>', unsafe_allow_html=True)
            st.plotly_chart(render_telemetry_breakdown_chart(fleet), use_container_width=True, key="breakdown")
    with b3:
        with st.container(border=True, height=250):
            st.markdown('<div class="panel-title">Battery state of charge</div>', unsafe_allow_html=True)
            st.plotly_chart(render_battery_chart(fleet), use_container_width=True, key="battery")
    with b4:
        with st.container(border=True, height=250):
            st.markdown('<div class="panel-title">Action point service times</div>', unsafe_allow_html=True)
            st.plotly_chart(render_service_times_chart(stats), use_container_width=True, key="service")

    with st.container(border=True, height=230):
        st.markdown('<div class="panel-title">Mission history</div>', unsafe_allow_html=True)
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, height=150, hide_index=True)
        else:
            st.info("Dispatch a mission from the sidebar to record live telemetry.")


# Streamlit >= 1.37: refresh only this fragment, so sidebar inputs are no longer
# re-run (and reset/flicker) every 200 ms. Older versions fall back to the old loop.
if hasattr(st, "fragment"):
    st.fragment(run_every=0.5)(live_view)()
else:
    live_view()
    time.sleep(0.5)
    st.rerun()