"""Streamlit front‑end for the multimodal data‑science engine.

Features required by the user (full‑screen dashboard with auto‑refresh, live event log,
chart gallery, and download links):

* **Session selector** – loads previous run directories from ``sessions/``.
* **Start / Stop** – spawns the orchestrator in a background thread.
* **Live bus view** – polls ``UIBusAdapter`` every few seconds.
* **Chart gallery** – renders Plotly figures from the ``VIZ_COMPLETE`` payloads.
* **Download** – provides buttons to download the raw log and each chart file.

The orchestrator already publishes ``SESSION_START`` and ``SESSION_END`` messages, so the UI can
track the current run ID and persist a log file under ``.session_logs``.
"""

import threading
import time
from pathlib import Path
from typing import List, Dict, Any

import sys
from pathlib import Path
# Add the repository root (two levels up) to PYTHONPATH so that `multimodal_ds` can be imported.
sys.path.append(str(Path(__file__).resolve().parents[2]))


import streamlit as st
from plotly.io import from_json as plotly_from_json

from multimodal_ds.graph import build_graph, make_initial_state
from multimodal_ds.config import OUTPUT_DIR

# ---------------------------------------------------------------------
# Helper: background worker that runs the LangGraph
# ---------------------------------------------------------------------

def _run_graph(
    file_paths: List[str],
    objective: str,
    session_id: str,
    stop_event: threading.Event,
) -> None:
    """Execute the full pipeline in a daemon thread."""
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        graph.invoke(
            make_initial_state(
                user_query=objective,
                uploaded_files=file_paths,
                session_id=session_id
            ),
            config=config
        )
    finally:
        stop_event.set()

# ---------------------------------------------------------------------
# Streamlit UI layout
# ---------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Multimodal DS Dashboard")

st.title("Multimodal Data‑Science Engine – Production Dashboard")

# ---------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------
session_root = OUTPUT_DIR
session_root.mkdir(parents=True, exist_ok=True)
existing_sessions = [p.name for p in session_root.iterdir() if p.is_dir()]

# Sidebar controls
theme_option = st.sidebar.selectbox("Theme", ["light", "dark"], index=1)
try:
    st.experimental_set_theme({"base": "dark" if theme_option == "dark" else "light"})
except Exception:
    pass

refresh_interval = st.sidebar.selectbox("Refresh interval (seconds)", [2, 5, 10], index=1)
st.session_state.refresh_interval = refresh_interval

selected_session = st.sidebar.selectbox(
    "Select a session",
    options=["<new run>"] + sorted(existing_sessions, reverse=True),
)

# ---------------------------------------------------------------------
# Controls for a new run
# ---------------------------------------------------------------------
if selected_session == "<new run>":
    file_paths_input = st.text_area(
        "File paths (comma‑separated)",
        placeholder="data/file1.csv, data/file2.pdf",
    )
    objective_input = st.text_input(
        "Objective",
        placeholder="Predict churn and explain key drivers",
    )
    start_button = st.button("Start Analysis")
else:
    start_button = False

# ---------------------------------------------------------------------
# Global objects
# ---------------------------------------------------------------------
if "orchestrator_thread" not in st.session_state:
    st.session_state.orchestrator_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# ---------------------------------------------------------------------
# Start a new run
# ---------------------------------------------------------------------
if start_button and file_paths_input and objective_input:
    new_session_id = f"session_{int(time.time())}"
    st.session_state.session_id = new_session_id
    st.session_state.stop_event.clear()
    
    thread = threading.Thread(
        target=_run_graph,
        args=(
            [p.strip() for p in file_paths_input.split(",") if p.strip()],
            objective_input,
            new_session_id,
            st.session_state.stop_event,
        ),
        daemon=True,
    )
    st.session_state.orchestrator_thread = thread
    thread.start()
    st.success(f"Analysis started – ID: {new_session_id}")

# ---------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------
current_sid = st.session_state.session_id if selected_session == "<new run>" else selected_session

if current_sid and current_sid != "<new run>":
    session_path = session_root / current_sid
    
    # 1. Executive Report
    report_file = session_path / "final_report.md"
    if report_file.exists():
        st.header("Executive Summary")
        st.markdown(report_file.read_text())
    else:
        if st.session_state.orchestrator_thread and st.session_state.orchestrator_thread.is_alive():
            st.info("Analysis in progress... Final report will appear here.")
        else:
            st.warning("No report found for this session.")

    # 2. Visualization Gallery
    st.divider()
    st.header("Visualizations")
    
    # Look for .json (Plotly) or .png files in the session directory
    viz_files = sorted(list(session_path.glob("*.json")) + list(session_path.glob("*.png")))
    if viz_files:
        cols = st.columns(2)
        for i, vfile in enumerate(viz_files):
            if vfile.name == "chart_manifest.json": continue
            
            with cols[i % 2]:
                with st.expander(f"View {vfile.name}", expanded=True):
                    if vfile.suffix == ".json":
                        try:
                            fig = plotly_from_json(vfile.read_text())
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.error(f"Error rendering {vfile.name}")
                    else:
                        st.image(str(vfile))
                    
                    with open(vfile, "rb") as f:
                        st.download_button(
                            f"Download {vfile.name}",
                            f,
                            file_name=vfile.name,
                            key=f"dl_{vfile.name}"
                        )
    else:
        st.write("No visualizations generated yet.")

# ---------------------------------------------------------------------
# Auto‑refresh
# ---------------------------------------------------------------------
if st.session_state.orchestrator_thread and st.session_state.orchestrator_thread.is_alive():
    if time.time() - st.session_state.last_refresh > st.session_state.refresh_interval:
        st.session_state.last_refresh = time.time()
        st.rerun()
