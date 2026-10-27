"""Data-science admin dashboard for all Part 2 experiments.

Run after the experiment scripts:   streamlit run dashboard.py
It reads each project's outputs/ folder (metrics.json, RESULTS.md, CSV tables, figures).
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
PROJECTS = {
    "01 · NYC Taxi Trip Duration": "01_nyc_taxi",
    "03 · Customer Segmentation": "03_customer_segmentation",
    "04 · Market Basket Mining": "04_market_basket",
    "06 · Anomaly Detection": "06_anomaly_detection",
    "12 · Time-Series Forecasting": "12_timeseries_forecasting",
}
st.set_page_config(page_title="CMPE 255 · DS Experiments", page_icon="🧪", layout="wide")
st.sidebar.title("🧪 DS Experiments")
st.sidebar.caption("Replication of dlmastery/data_science_examples · CRISP-DM")
choice = st.sidebar.radio("Project", list(PROJECTS))
out = BASE / PROJECTS[choice] / "outputs"

st.title(choice)
if not (out / "metrics.json").exists():
    st.warning(f"No results yet. Run `python {PROJECTS[choice]}/*.py` first.")
    st.stop()

metrics = json.loads((out / "metrics.json").read_text())
c1, c2 = st.columns([2, 1])
with c2:
    st.metric("Runtime (s)", metrics.get("runtime_seconds", "–"))
    with st.expander("Raw metrics.json"):
        st.json(metrics)
with c1:
    st.markdown((out / "RESULTS.md").read_text())

st.header("Figures")
for f in sorted((out / "figures").glob("*.png")):
    st.subheader(f.stem.split("_", 1)[-1].replace("_", " ").title())
    st.image(str(f))

st.header("Tables")
for f in sorted(out.glob("*.csv")):
    with st.expander(f.name):
        st.dataframe(pd.read_csv(f))
