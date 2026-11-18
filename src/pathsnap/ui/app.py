import os
import requests
import streamlit as st
import streamlit.components.v1 as components

from src.pathsnap.schemas import DecodedPoint
from visualizations.maps import build_trip_folium_map

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap Evaluator UI", layout="wide")
st.title("PathSnap: Distributed Map-Matching & Route Reconstruction")
st.caption(f"Connected API Endpoint: `{API_URL}`")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    trips_resp = requests.get(f"{API_URL}/trips", timeout=3).json()
    st.sidebar.success(f"API Online ({health.get('artifact_version')})")
    trips = trips_resp.get("trips", [])
except Exception as exc:
    st.sidebar.error(f"Cannot reach API at {API_URL}: {exc}")
    st.stop()

selected_trip = st.sidebar.selectbox("Select Evaluation Trip", trips)
if st.sidebar.button("Run Route Reconstruction", type="primary"):
    with st.spinner(f"Running inference on {selected_trip}..."):
        res = requests.post(
            f"{API_URL}/predict", json={"trip_id": selected_trip}, timeout=30
        )
        if res.status_code != 200:
            st.error(f"API Error {res.status_code}: {res.text}")
            st.stop()
        st.session_state["last_result"] = res.json()

data = st.session_state.get("last_result")
if data:
    st.subheader(
        f"Reconstructed Route — {data['trip_id']} (Req ID: {data['request_id']})"
    )

    for w in data.get("warnings", []):
        st.warning(w)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    metrics = data.get("metrics") or {}
    timings = data.get("timings_ms") or {}

    top1 = metrics.get("observation_top1_accuracy")
    seq_sim = metrics.get("sequence_similarity")

    m_col1.metric("Top-1 Accuracy", top1 if top1 is not None else "N/A")
    m_col2.metric("Route Continuity", metrics.get("route_continuity", "N/A"))
    m_col3.metric("Sequence Similarity", seq_sim if seq_sim is not None else "N/A")
    m_col4.metric("Total Latency (ms)", timings.get("total", "N/A"))

    decoded_objs = [
        DecodedPoint(
            observation_id=pt["observation_id"],
            timestamp=pt["timestamp"],
            gps_lon=pt["gps_lon"],
            gps_lat=pt["gps_lat"],
            arc_id=pt["arc_id"],
            proj_fraction=pt["proj_fraction"],
            status=pt["status"],
        )
        for pt in data.get("points", [])
    ]

    fmap = build_trip_folium_map(
        points=decoded_objs,
        predicted_segments=data.get("route_geometry", []),
    )
    components.html(fmap._repr_html_(), height=500, scrolling=False)

    col1, col2 = st.columns(2)
    with col1:
        st.json({"metrics": metrics, "timings_ms": timings})
    with col2:
        st.json({"predicted_arcs": data.get("predicted_arcs")})