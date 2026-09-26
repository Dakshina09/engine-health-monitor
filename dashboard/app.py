"""Engine Health Monitor: Streamlit entry point.

    streamlit run dashboard/app.py

Sets the favicon and page title, registers the pages (dashboard, privacy
policy, terms and conditions) and renders the site footer on every page.
The dashboard talks to the FastAPI backend at $API_URL (default
http://localhost:8000) and falls back to running the model in-process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from legal import privacy_policy, terms_and_conditions  # noqa: E402
from site_config import SITE_NAME  # noqa: E402

st.set_page_config(page_title=SITE_NAME, page_icon=str(HERE / "static" / "favicon.png"), layout="wide")

pages = [
    st.Page(str(HERE / "monitor.py"), title="Dashboard", url_path="dashboard", default=True),
    st.Page(privacy_policy, title="Privacy policy", url_path="privacy"),
    st.Page(terms_and_conditions, title="Terms and conditions", url_path="terms"),
]
current = st.navigation(pages, position="hidden")
current.run()

# ------------------------------------------------------------------ footer
st.divider()
cols = st.columns([4, 1, 1, 1])
cols[0].caption(f"{SITE_NAME}. Built on the NASA C-MAPSS simulated turbofan dataset.")
cols[1].page_link(pages[0], label="Dashboard")
cols[2].page_link(pages[1], label="Privacy policy")
cols[3].page_link(pages[2], label="Terms")
