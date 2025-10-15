import streamlit as st
from stfu.state import init_state
from stfu.ui.pages import setup, class_session

st.set_page_config(page_title="Quiet Break Timer", page_icon="🥛", layout="centered")

# Initialize session state keys (idempotent)
init_state()

# Simple router
if not st.session_state.configured:
    setup.render()
else:
    class_session.render()
