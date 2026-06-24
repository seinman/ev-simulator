import streamlit as st

from ev_simulator.dashboard.charts import (
    flexibility_fig,
    kwh_topped_fig,
    plug_in_out_heatmap_fig,
    plug_in_rate_fig,
    plug_in_soc_fig,
    soc_fig,
)
from ev_simulator.dashboard.compute import compute_metrics, compute_session_metrics
from ev_simulator.simulation import Simulator

st.set_page_config(
    page_title="EV Flexibility Simulator",
    page_icon="⚡",
    layout="wide",
)

st.title("EV Flexibility Simulator")
st.caption("Simulates a representative week of EV charging behaviour across the GB fleet.")

# ── Sidebar: simulation controls ──────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation")
    n_users = st.slider("Fleet size", min_value=1000, max_value=10000, value=5000, step=500)
    seed    = st.number_input("Random seed", value=42, min_value=0, step=1)
    season  = st.selectbox("Price week season", ["Winter", "Spring", "Summer", "Autumn"])
    run     = st.button("Run simulation", type="primary", use_container_width=True)

    st.divider()
    st.header("Filters")

# ── Run simulation and cache result ───────────────────────────────────────────
if run:
    with st.spinner("Running simulation…"):
        try:
            sim = Simulator(n_users=int(n_users), seed=int(seed), season=season)
            st.session_state["result"] = sim.run()
        except (RuntimeError, ValueError) as e:
            st.error(str(e))
            st.stop()

if "result" not in st.session_state:
    st.info("Set parameters in the sidebar and click **Run simulation** to begin.")
    st.stop()

result = st.session_state["result"]

# ── Archetype filter (operates on cached result — no re-simulation needed) ────
all_archetypes = sorted({u.archetype.name for u in result.users})
with st.sidebar:
    selected = st.multiselect("Archetypes", options=all_archetypes, default=all_archetypes)

if not selected:
    st.warning("Select at least one archetype to display results.")
    st.stop()

archetype_filter = set(selected)

try:
    metrics = compute_metrics(result, archetype_filter)
    session = compute_session_metrics(result, archetype_filter)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Users (filtered)",      f"{metrics.n_users:,}")
c2.metric("Peak plug-in rate",     f"{metrics.peak_plug_in_pct:.1f}%")
c3.metric("Peak charging demand",  f"{metrics.peak_charging_mw:.1f} MW")
c4.metric("Peak available flex",   f"{metrics.peak_flex_mw:.1f} MW")

# ── Fleet overview charts ─────────────────────────────────────────────────────
col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(plug_in_rate_fig(metrics), use_container_width=True)
with col_r:
    st.plotly_chart(soc_fig(metrics), use_container_width=True)

st.plotly_chart(flexibility_fig(metrics), use_container_width=True)

# ── Session statistics ────────────────────────────────────────────────────────
st.divider()
st.subheader("Charging session statistics")

st.plotly_chart(plug_in_soc_fig(session), use_container_width=True)
st.plotly_chart(kwh_topped_fig(session), use_container_width=True)
st.plotly_chart(plug_in_out_heatmap_fig(session), use_container_width=True)
