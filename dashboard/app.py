"""Performance dashboard.

Run with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect.changepoint import find_changepoint  # noqa: E402
from detect.regression import DEFAULTS, score_series  # noqa: E402

DB_PATH = "results/perf.duckdb"

st.set_page_config(page_title="ml-perf-sentinel", layout="wide")
st.title("ml-perf-sentinel — Performance & Release Readiness")

if not Path(DB_PATH).exists():
    st.warning("No warehouse found. Run: `python scripts/seed_history.py --inject mlp` "
               "then `python scripts/run_pipeline.py`")
    st.stop()

con = duckdb.connect(DB_PATH, read_only=True)

combos = con.execute(
    "SELECT DISTINCT workload, mode, batch_size FROM benchmarks ORDER BY 1, 2, 3"
).fetchall()
metrics = [r[0] for r in con.execute("SELECT DISTINCT metric FROM benchmarks").fetchall()]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Series")
workload = st.sidebar.selectbox("Workload", sorted({c[0] for c in combos}))
mode = st.sidebar.selectbox("Mode", sorted({c[1] for c in combos if c[0] == workload}))
batch_size = st.sidebar.selectbox(
    "Batch size", sorted({c[2] for c in combos if c[0] == workload and c[1] == mode}))
metric = st.sidebar.selectbox(
    "Metric", metrics, index=metrics.index("latency_ms_p50") if "latency_ms_p50" in metrics else 0)

# ------------------------------------------------------- release readiness
def series_df(w, m, bs, met) -> pd.DataFrame:
    return con.execute(
        """
        SELECT run_id, min(ts) AS ts, median(value) AS value
        FROM benchmarks
        WHERE workload = ? AND mode = ? AND batch_size = ? AND metric = ?
        GROUP BY run_id ORDER BY ts
        """, [w, m, bs, met]).df()

open_regressions = []
for w, m, bs in combos:
    df_ = series_df(w, m, bs, DEFAULTS["metric"])
    s = score_series(df_["value"].tolist())
    if s["is_regression"]:
        open_regressions.append((w, m, bs, s))

left, right = st.columns([1, 3])
with left:
    if open_regressions:
        st.error(f"Release gate: RED — {len(open_regressions)} open regression(s)")
    else:
        st.success("Release gate: GREEN — no open regressions")
with right:
    for w, m, bs, s in open_regressions:
        st.markdown(f"- :red[**{w}** / {m} / bs={bs}] — {s['pct_change']:+.1f}% "
                    f"(z={s['z']:.1f}, MW p={s['mw_pvalue']:.2g})")

st.divider()

# --------------------------------------------------------------- trend plot
df = series_df(workload, mode, batch_size, metric)
score = score_series(df["value"].tolist())
cp = find_changepoint(df["value"].tolist())

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["ts"], y=df["value"], mode="lines+markers",
                         name=metric, line=dict(width=2)))

if len(df) > DEFAULTS["baseline_window"]:
    baseline = df["value"].iloc[-(DEFAULTS["baseline_window"] + 1):-1]
    fig.add_hline(y=baseline.mean(), line_dash="dash", annotation_text="baseline mean")

if score["is_regression"]:
    fig.add_trace(go.Scatter(x=[df["ts"].iloc[-1]], y=[df["value"].iloc[-1]],
                             mode="markers", name="regression",
                             marker=dict(color="red", size=14, symbol="x")))
if cp.get("index") is not None and score["is_regression"]:
    fig.add_vline(x=df["ts"].iloc[cp["index"]], line_color="red", line_dash="dot",
                  annotation_text=f"changepoint ({df['run_id'].iloc[cp['index']]})")

fig.update_layout(height=420, margin=dict(t=30), xaxis_title="run time",
                  yaxis_title=metric)
st.subheader(f"{workload} / {mode} / batch={batch_size}")
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest", f"{df['value'].iloc[-1]:.3f}")
c2.metric("vs baseline", f"{score['pct_change']:+.1f}%" if score["pct_change"] is not None else "n/a")
c3.metric("z-score", f"{score['z']:.2f}" if score["z"] is not None else "n/a")
c4.metric("Mann-Whitney p", f"{score['mw_pvalue']:.3g}" if score["mw_pvalue"] is not None else "n/a")

# --------------------------------------------------- eager vs compiled view
st.divider()
st.subheader("Compilation speedup (eager vs compiled)")
comp = con.execute(
    """
    WITH latest AS (SELECT max(ts) AS ts FROM benchmarks)
    SELECT workload, batch_size,
           median(value) FILTER (mode = 'eager')    AS eager_ms,
           median(value) FILTER (mode = 'compiled') AS compiled_ms
    FROM benchmarks, latest
    WHERE metric = 'latency_ms_p50' AND benchmarks.ts = latest.ts
    GROUP BY workload, batch_size ORDER BY workload, batch_size
    """).df()
if not comp.empty and comp["compiled_ms"].notna().any():
    comp["speedup"] = (comp["eager_ms"] / comp["compiled_ms"]).round(2)
    st.dataframe(comp, use_container_width=True, hide_index=True)
else:
    st.caption("No compiled-mode data in the latest run (torch backend required).")

con.close()
