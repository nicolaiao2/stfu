import streamlit as st
from ...state import init_state

def render():
    init_state()
    st.title("🥛 STFU - Stille Tid For Undervisning")
    st.caption("Sett varighet, maks pause og straff. Klar? Trykk start.")
    ss = st.session_state
    ss.class_minutes = st.number_input("Undervisningstid (minutter)", 1, 240, int(ss.class_minutes))
    ss.max_break_minutes = float(st.number_input("Maks pausetid (min) / startvolum (ml)", 1, 1000, int(ss.max_break_minutes)))
    ss.initial_water_ml = ss.max_break_minutes
    ss.penalty_pct = float(st.slider("Høylytt straff (% av gjenværende vann som søles)", 1.0, 30.0, float(ss.penalty_pct), 1.0))
    st.write("---")
    if st.button("Start timen", type="primary"):
        ss.configured = True
        ss.started_at = None
        ss.last_update = None
        ss.timer_running = False
        ss.water_left_ml = float(ss.initial_water_ml)
        ss.done = False
        st.rerun()
