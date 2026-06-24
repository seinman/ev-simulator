from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ev_simulator.dashboard.compute import FleetMetrics, SessionMetrics

_BLUE   = "#4c84b0"
_ORANGE = "#e07b39"
_GREEN  = "#3cb371"


def plug_in_rate_fig(metrics: FleetMetrics) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=metrics.bloc_labels,
        y=metrics.plug_in_rate * 100,
        marker_color=_BLUE,
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Fleet plug-in rate",
        xaxis_title="Hour of day",
        yaxis=dict(title="%", range=[0, 100]),
        height=360,
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def soc_fig(metrics: FleetMetrics) -> go.Figure:
    x = metrics.bloc_labels

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=np.concatenate([metrics.p75_soc, metrics.p25_soc[::-1]]) * 100,
        fill="toself",
        fillcolor=f"rgba(60,179,113,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IQR (25–75%)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=metrics.mean_soc * 100,
        line=dict(color=_GREEN, width=2.5),
        name="Mean SoC",
        hovertemplate="%{x}: %{y:.1f}%<extra>Mean SoC</extra>",
    ))
    fig.add_hline(
        y=80, line_dash="dash", line_color="tomato", opacity=0.6,
        annotation_text="Target SoC (80%)", annotation_position="top right",
    )
    fig.update_layout(
        title="SoC distribution — plugged-in users (mean ± IQR)",
        xaxis_title="Hour of day",
        yaxis=dict(title="%", range=[0, 100]),
        height=360,
        margin=dict(t=50, b=50, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def flexibility_fig(metrics: FleetMetrics) -> go.Figure:
    x = metrics.bloc_labels

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=metrics.fleet_charging_kw / 1000,
        name="Charging now (MW)",
        marker_color=_ORANGE,
        hovertemplate="%{x}: %{y:.2f} MW<extra>Charging</extra>",
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=metrics.available_flex_kw / 1000,
        name="Dispatchable headroom (MW)",
        marker_color=_GREEN,
        opacity=0.8,
        hovertemplate="%{x}: %{y:.2f} MW<extra>Available flex</extra>",
    ))
    fig.update_layout(
        title="Fleet charging demand and available flexibility",
        xaxis_title="Hour of day",
        yaxis_title="MW",
        barmode="stack",
        height=400,
        margin=dict(t=50, b=50, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plug_in_soc_fig(session: SessionMetrics) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=session.plug_in_soc * 100,
        nbinsx=40,
        marker_color=_BLUE,
        opacity=0.85,
        hovertemplate="SoC: %{x:.0f}%<br>Events: %{y}<extra></extra>",
    ))
    fig.add_vline(x=80, line_dash="dash", line_color="tomato", opacity=0.7,
                  annotation_text="80% cap", annotation_position="top right")
    fig.update_layout(
        title="Plug-in SoC at connection",
        xaxis_title="SoC (%)",
        yaxis_title="Sessions",
        height=360,
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def kwh_topped_fig(session: SessionMetrics) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=session.kwh_topped,
        nbinsx=40,
        marker_color=_ORANGE,
        opacity=0.85,
        hovertemplate="kWh: %{x:.1f}<br>Sessions: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Energy added per charging session",
        xaxis_title="kWh added",
        yaxis_title="Sessions",
        height=360,
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def plug_in_out_heatmap_fig(session: SessionMetrics) -> go.Figure:
    h, xedges, yedges = np.histogram2d(
        session.plug_in_hod, session.plug_out_hod,
        bins=24, range=[[0, 24], [0, 24]], density=True,
    )
    fig = go.Figure(go.Heatmap(
        x=xedges,
        y=yedges,
        z=h.T,
        colorscale="YlOrRd",
        hovertemplate="Plug-in: %{x:.1f}h<br>Plug-out: %{y:.1f}h<br>Density: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Density"),
    ))
    fig.update_layout(
        title="Plug-in / plug-out time density",
        xaxis=dict(title="Plug-in hour of day", tickvals=list(range(0, 25, 4))),
        yaxis=dict(title="Plug-out hour of day", tickvals=list(range(0, 25, 4))),
        height=360,
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig
