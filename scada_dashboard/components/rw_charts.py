"""Plotly figures for the RoboWeave dashboard (dark theme matching the old SCADA look)."""
import math

import plotly.graph_objects as go

BG, PANEL, LINE = "#0d1b2a", "#0a1724", "#1b3348"
TEXT, MUTED = "#d6e4f0", "#7f98ae"
CYAN, GREEN, AMBER, RED, BLUE = "#22d3ee", "#34d399", "#f59e0b", "#ef4444", "#60a5fa"
STATE_COLOR = {"GO": CYAN, "slowed": "#38bdf8", "YIELD": AMBER, "STOP": RED, "WAIT": "#a78bfa", "DONE": GREEN}
MODE_LABEL = {"roboweave": "RoboWeave", "box": "Lidar-only: route around", "stop_all": "Lidar-only: stop & wait"}
MODE_COLOR = {"roboweave": CYAN, "box": "#94a3b8", "stop_all": "#f87171"}


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11), showlegend=False)
    return fig


def _circle(cx, cy, r, **kw):
    return dict(type="circle", xref="x", yref="y", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r, **kw)


def floor_map(s, height=430):
    w = s["world"]
    fig = go.Figure()
    shapes = [dict(type="rect", x0=0, y0=0, x1=w["width"], y1=w["height"], line=dict(color=LINE, width=2),
                   fillcolor="#081420", layer="below")]
    for x, y, rw, rh in w["racks"]:
        shapes.append(dict(type="rect", x0=x, y0=y, x1=x + rw, y1=y + rh, line=dict(color="#b8732e", width=1),
                           fillcolor="#7a4d22"))
    for x, y, rw, rh in w["crates"]:
        shapes.append(dict(type="rect", x0=x, y0=y, x1=x + rw, y1=y + rh, line=dict(color="#e2a15a", width=1.5),
                           fillcolor="#a8743a"))
    for x, y, r in w["pillars"]:
        shapes.append(_circle(x, y, r, line=dict(color="#cbd5e1", width=1), fillcolor="#8b97a6"))
    z = s["yield_zone"]
    for x, y, r in s["people"]:
        shapes.append(_circle(x, y, z, line=dict(color=AMBER, width=1.2, dash="dot"), fillcolor="rgba(245,158,11,0.07)"))
        shapes.append(_circle(x, y, r, line=dict(color="#fde68a", width=1), fillcolor="#facc15"))
    for o in s["perceived"]:
        col = GREEN if o["cls"] == "person" else BLUE
        shapes.append(_circle(o["x"], o["y"], 0.45, line=dict(color=col, width=2, dash="dash" if o["unsure"] else "solid")))
    rb = s["robot"]
    shapes.append(_circle(rb["x"], rb["y"], rb["r"], line=dict(color="#a5f3fc", width=1), fillcolor=CYAN))
    fig.update_layout(shapes=shapes)
    if s["path"]:
        xs = [rb["x"]] + [p[0] for p in s["path"]]
        ys = [rb["y"]] + [p[1] for p in s["path"]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=CYAN, width=2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[rb["x"], rb["x"] + 0.6 * math.cos(rb["theta"])],
                             y=[rb["y"], rb["y"] + 0.6 * math.sin(rb["theta"])], mode="lines",
                             line=dict(color="#ffffff", width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[s["goal"][0]], y=[s["goal"][1]], mode="markers+text", text=["goal"], textposition="top center",
                             marker=dict(symbol="x", size=12, color=RED), textfont=dict(color=RED, size=10), hoverinfo="skip"))
    fig.update_xaxes(range=[-0.3, w["width"] + 0.3], visible=False, constrain="domain")
    fig.update_yaxes(range=[-0.3, w["height"] + 0.3], visible=False, scaleanchor="x", scaleratio=1)
    return _base(fig, height)


def state_donut(state_time, height=210):
    items = [(k, v) for k, v in state_time.items() if k not in ("DONE", "slowed") and v > 0]
    fig = go.Figure()
    if items:
        fig.add_trace(go.Pie(labels=[k for k, _ in items], values=[v for _, v in items], hole=0.62, sort=False,
                             marker=dict(colors=[STATE_COLOR.get(k, MUTED) for k, _ in items], line=dict(color=BG, width=2)),
                             textinfo="label+percent", textfont=dict(size=11), hovertemplate="%{label}: %{value:.1f} s<extra></extra>"))
    return _base(fig, height)


def speed_trace(trace, height=200):
    fig = go.Figure()
    if trace:
        t = [r[0] for r in trace]
        fig.add_trace(go.Scatter(x=t, y=[r[1] for r in trace], mode="lines", line=dict(color=CYAN, width=2, shape="hv"),
                                 fill="tozeroy", fillcolor="rgba(34,211,238,0.12)", hovertemplate="%{x:.1f}s  %{y:.2f} m/s<extra></extra>"))
    fig.update_xaxes(gridcolor=LINE, title=dict(text="sim time (s)", font=dict(size=10, color=MUTED)))
    fig.update_yaxes(gridcolor=LINE, range=[0, 1.2], title=dict(text="speed (m/s)", font=dict(size=10, color=MUTED)))
    return _base(fig, height)


def gap_trace(trace, height=200):
    fig = go.Figure()
    pts = [(r[0], r[2]) for r in trace if r[2] is not None]
    if pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[min(p[1], 8) for p in pts], mode="lines",
                                 line=dict(color=AMBER, width=2), hovertemplate="%{x:.1f}s  %{y:.2f} m<extra></extra>"))
    else:
        fig.add_annotation(text="no people in this scenario", showarrow=False, font=dict(color=MUTED))
    fig.update_xaxes(gridcolor=LINE, title=dict(text="sim time (s)", font=dict(size=10, color=MUTED)))
    fig.update_yaxes(gridcolor=LINE, rangemode="tozero", title=dict(text="gap to nearest person (m)", font=dict(size=10, color=MUTED)))
    return _base(fig, height)


def compare_time(results, timeout=60.0, height=300):
    scen = list(dict.fromkeys(r["scenario"] for r in results))
    fig = go.Figure()
    for mode in ("roboweave", "box", "stop_all"):
        by = {r["scenario"]: r for r in results if r["mode"] == mode}
        ys = [(by[s]["time"] or timeout) if s in by else None for s in scen]
        txt = [("did not finish" if s in by and not by[s]["time"] else f"{by[s]['time']:.1f}") if s in by else "" for s in scen]
        fig.add_trace(go.Bar(name=MODE_LABEL[mode], x=scen, y=ys, text=txt, textposition="outside",
                             marker=dict(color=MODE_COLOR[mode]), textfont=dict(size=10)))
    fig.update_layout(barmode="group", showlegend=True, legend=dict(orientation="h", y=1.18, font=dict(size=11, color=TEXT)))
    fig.update_yaxes(gridcolor=LINE, title=dict(text="time to goal (s)", font=dict(size=10, color=MUTED)), range=[0, timeout * 1.15])
    return _base(fig, height).update_layout(showlegend=True, margin=dict(l=8, r=8, t=34, b=8))


def compare_gap(results, height=300):
    scen = list(dict.fromkeys(r["scenario"] for r in results if r["min_gap"] is not None))
    fig = go.Figure()
    for mode in ("roboweave", "box", "stop_all"):
        by = {r["scenario"]: r for r in results if r["mode"] == mode}
        ys = [by[s]["min_gap"] if s in by else None for s in scen]
        fig.add_trace(go.Bar(name=MODE_LABEL[mode], x=scen, y=ys, text=[f"{y:.2f}" if y is not None else "" for y in ys],
                             textposition="outside", marker=dict(color=MODE_COLOR[mode]), textfont=dict(size=10)))
    fig.update_layout(barmode="group", showlegend=True, legend=dict(orientation="h", y=1.18, font=dict(size=11, color=TEXT)))
    fig.update_yaxes(gridcolor=LINE, title=dict(text="closest approach to a person (m)", font=dict(size=10, color=MUTED)))
    return _base(fig, height).update_layout(showlegend=True, margin=dict(l=8, r=8, t=34, b=8))