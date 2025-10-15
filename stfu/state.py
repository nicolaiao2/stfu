import streamlit as st
from stfu.audio.sound import load_sound_b64

DEFAULT_THRESHOLD = 1.0
SOUND_FILE = "sounds/hey.mp3"  # existing chosen file

def init_state():
    ss = st.session_state
    defaults = {
        'configured': False,
        'class_minutes': 45,
        'initial_water_ml': 45.0,  # 1 ml = 1 break min
        'max_break_minutes': 15.0,
        'started_at': None,
        'last_update': None,
        'water_left_ml': 45.0,
        'done': False,
        'max_rms': 0.0,
        'rms_history': [],
        'noise_floor_rms': 0.0,
        'smoothed_level': 0.0,
        'smoothing_alpha': 0.3,
        'debug_override_enabled': False,
        'debug_level_override': 0.0,
        'penalty_pct': 5.0,
        'tilt_until': 0.0,
        'last_penalty_time': 0.0,
        'penalty_events': 0,
        'last_penalty_intensity': 0.0,
        'last_sound_played_time': 0.0,
        'noise_threshold': DEFAULT_THRESHOLD,
        'timer_running': False,
    }
    for k,v in defaults.items():
        ss.setdefault(k, v)
    if 'alert_sound_b64' not in ss:
        ss.alert_sound_b64 = load_sound_b64(SOUND_FILE)
