import plotly.graph_objects as go

# ---- Palette (matches the CSS tokens in dashboard.py) ----
BG = "rgba(0,0,0,0)"
GRID = "#16293b"
TEXT = "#8fa8bf"
CYAN = "#22d3ee"
TEAL = "#2dd4bf"
GREEN = "#34d399"
AMBER = "#f59e0b"
RED = "#ef4444"
SHELF_FILL = "rgba(30,64,98,0.55)"
SHELF_LINE = "#2b5a85"

STATUS_COLOR = {
    "NAVIGATING": CYAN,
    "COMPLETED": GREEN,
    "STANDBY": GREEN,
    "BLOCKED": RED,
}

LAYOUT_BASE = dict(
    margin=dict(l=6, r=6, t=6, b=6),
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    font=dict(family="IBM Plex Sans, Inter, sans-serif", size=11, color=TEXT),
)


def _layout(top=6):
    d = dict(LAYOUT_BASE)
    d["margin"] = dict(l=6, r=10, t=top, b=6)
    return d


BOTTOM_TOP = 34  # same top margin on every bottom-row chart so plot areas line up


def _axes(**kw):
    base = dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)
    base.update(kw)
    return base


def render_floorplan_chart(grid, fleet, height=410):
    gh, gw = len(grid), len(grid[0])
    fig = go.Figure()

    # Shelves: ONE trace for every obstacle cell (was one shape per cell)
    sx, sy = [], []
    for r in range(gh):
        for c in range(gw):
            if grid[r][c] == 1:
                sx += [c - 0.42, c + 0.42, c + 0.42, c - 0.42, c - 0.42, None]
                sy += [r - 0.42, r - 0.42, r + 0.42, r + 0.42, r - 0.42, None]
    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode="lines", fill="toself", fillcolor=SHELF_FILL,
        line=dict(color=SHELF_LINE, width=1), hoverinfo="skip", showlegend=False))

    # Lanes: ONE trace for every free-cell connection (was one trace per segment)
    lx, ly = [], []
    for r in range(gh):
        for c in range(gw):
            if grid[r][c] == 0:
                if c + 1 < gw and grid[r][c + 1] == 0:
                    lx += [c, c + 1, None]
                    ly += [r, r, None]
                if r + 1 < gh and grid[r + 1][c] == 0:
                    lx += [c, c, None]
                    ly += [r, r + 1, None]
    fig.add_trace(go.Scatter(
        x=lx, y=ly, mode="lines", line=dict(color="#183a52", width=1.5),
        hoverinfo="skip", showlegend=False))

    for r_id, amr in fleet.items():
        col = STATUS_COLOR.get(amr["status"], CYAN)

        if amr.get("path"):
            px = [p[1] for p in amr["path"]]
            py = [p[0] for p in amr["path"]]
            # glow (wide, faint) + core (thin, bright)
            fig.add_trace(go.Scatter(x=px, y=py, mode="lines",
                                     line=dict(color=col, width=10), opacity=0.14,
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=px, y=py, mode="lines",
                                     line=dict(color=col, width=2.5), opacity=0.9,
                                     hoverinfo="skip", showlegend=False))

        # Destination ring
        fig.add_trace(go.Scatter(
            x=[amr["goal"][1]], y=[amr["goal"][0]], mode="markers+text",
            marker=dict(symbol="circle-open", size=16, color=AMBER, line=dict(width=2.5)),
            text=[f"{r_id} dest"], textposition="top center",
            textfont=dict(size=9, color=AMBER), hoverinfo="skip", showlegend=False))

        # Robot marker with a soft halo
        cy, cx = amr["pos"]
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                                 marker=dict(symbol="circle", size=34, color=col), opacity=0.18,
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers+text",
            marker=dict(symbol="square", size=20, color="#07111b", line=dict(color=col, width=2.5)),
            text=[r_id], textposition="bottom center",
            textfont=dict(size=10, color="#e6f1fa"),
            hovertemplate=f"{r_id}<br>Status: {amr['status']}<br>Battery: {amr['battery']:.0f}%<extra></extra>",
            showlegend=False))

    fig.update_layout(
        **LAYOUT_BASE, height=height,
        xaxis=dict(range=[-0.6, gw - 0.4], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[gh - 0.4, -0.6], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def render_telemetry_breakdown_chart(fleet, height=180):
    robots = list(fleet.keys())
    parts = {"Cargo": ("cargo", TEAL), "Empty": ("empty", "#3b82f6"),
             "Docking": ("docking", AMBER), "Blocked": ("blocked", RED)}
    fig = go.Figure()
    for label, (key, color) in parts.items():
        vals = []
        for r in robots:
            t = fleet[r]["ticks"]
            vals.append(round(t[key] / max(1, sum(t.values())) * 100, 1))
        fig.add_trace(go.Bar(y=robots, x=vals, orientation="h", name=label,
                             marker=dict(color=color)))
    fig.update_layout(**_layout(BOTTOM_TOP), barmode="stack", height=height,
                      xaxis=_axes(range=[0, 100]), yaxis=dict(tickfont=dict(size=11)),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom",
                                  font=dict(size=10), traceorder="normal"))
    return fig


def render_battery_chart(fleet, height=180):
    robots = list(fleet.keys())
    vals = [fleet[r]["battery"] for r in robots]
    colors = [RED if v < 20 else AMBER if v < 50 else GREEN for v in vals]
    fig = go.Figure(go.Bar(
        y=robots, x=vals, orientation="h", marker=dict(color=colors),
        text=[f"{v:.0f}%" for v in vals], textposition="inside",
        textfont=dict(color="#07111b", size=11)))
    fig.update_layout(**_layout(BOTTOM_TOP), height=height, xaxis=_axes(range=[0, 100]),
                      yaxis=dict(tickfont=dict(size=11)), showlegend=False)
    return fig


def render_service_times_chart(action_point_stats, height=180):
    pts = list(action_point_stats.keys())
    fig = go.Figure(go.Bar(y=pts, x=list(action_point_stats.values()), orientation="h",
                           marker=dict(color=CYAN)))
    fig.update_layout(**_layout(BOTTOM_TOP), height=height, xaxis=_axes(title="seconds"),
                      yaxis=dict(tickfont=dict(size=10)), showlegend=False)
    return fig


def render_output_chart(buckets, height=180):
    """buckets: list of (label, count) - deliveries per minute."""
    fig = go.Figure()
    if buckets:
        fig.add_trace(go.Scatter(
            x=[b[0] for b in buckets], y=[b[1] for b in buckets], mode="lines+markers",
            line=dict(color=CYAN, width=2.5, shape="spline"),
            marker=dict(size=6, color=CYAN),
            fill="tozeroy", fillcolor="rgba(34,211,238,0.15)"))
    else:
        fig.add_annotation(text="Deliveries will appear here as missions complete",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           font=dict(color=TEXT, size=12))
    fig.update_layout(**_layout(BOTTOM_TOP), height=height, showlegend=False,
                      xaxis=_axes(showgrid=False), yaxis=_axes(rangemode="tozero", dtick=1))
    return fig


def render_health_gauge(pct, height=140):
    color = GREEN if pct >= 75 else AMBER if pct >= 40 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", font=dict(size=34, color="#e6f1fa")),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=GRID, tickfont=dict(size=9)),
                   bar=dict(color=color, thickness=0.28), bgcolor="#0d1b2a", borderwidth=0,
                   steps=[dict(range=[0, 100], color="#10263a")])))
    fig.update_layout(**LAYOUT_BASE, height=height)
    return fig