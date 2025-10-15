import time, math
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ...state import init_state
from ...audio.processor import NoiseProcessor
from ..glass import render_glass

def render():
    init_state()
    st.title("🥛 Stillepause – I timen")
    st.caption("Blir det bråk lekker glasset! Vann igjen = pausetid.")
    ss = st.session_state

    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
        audio_processor_factory=NoiseProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    ml_total = float(ss.initial_water_ml)
    class_secs = float(ss.class_minutes) * 60.0
    baseline_rate_per_sec = ml_total / class_secs if class_secs > 0 else 0.0

    now = time.time()
    if ss.timer_running:
        dt = 0.0 if ss.last_update is None else max(0.0, now - ss.last_update)
        ss.last_update = now
    else:
        dt = 0.0

    display_level = 0.0
    raw_rms = 0.0
    processor_status = "not-created"
    manual_frames_used = False

    if ctx is not None and getattr(ctx.state, 'playing', False):
        if ctx.audio_processor is not None:
            processor_status = "ready"
            raw_rms = float(getattr(ctx.audio_processor, 'raw_rms', 0.0))
        else:
            processor_status = "missing (fallback active)"
            receiver = getattr(ctx, 'audio_receiver', None)
            if receiver is not None:
                try:
                    frames = receiver.get_frames(timeout=0.1)
                    import av  # type: ignore
                    levels = []
                    for f in frames:
                        arr = f.to_ndarray()
                        if arr.ndim == 2:
                            arr = arr.mean(axis=0)
                        x = arr.astype(np.float32)
                        if x.size > 0 and np.max(np.abs(x)) > 2.0:
                            x = x / 32768.0
                        x = np.clip(x, -1.0, 1.0)
                        lvl = float(np.sqrt(np.mean(x**2)) + 1e-12)
                        levels.append(lvl)
                    if levels:
                        raw_rms = float(np.mean(levels))
                        manual_frames_used = True
                except Exception as e:
                    print(f"Manual frame pull failed: {e}")
    elif ctx is not None:
        processor_status = "connecting"

    if processor_status != 'ready':
        info_msg = f"Lydstatus: {processor_status}. Tillat mikrofon og vent litt."
        if manual_frames_used:
            info_msg += " (midlertidig reserve aktiv)."
        st.info(info_msg)

    if processor_status in ('ready', 'missing (fallback active)'):
        nf = ss.noise_floor_rms
        if nf == 0.0:
            ss.noise_floor_rms = raw_rms
        else:
            decay_alpha = 0.05 if raw_rms < nf * 1.5 else 0.001
            ss.noise_floor_rms = (1 - decay_alpha) * nf + decay_alpha * raw_rms
        effective = max(0.0, raw_rms - ss.noise_floor_rms * 1.2)
        compressed = math.pow(effective, 0.5)
        unsmoothed = min(1.2, compressed * 3.0)
        alpha = float(ss.smoothing_alpha)
        ss.smoothed_level = unsmoothed if ss.smoothed_level == 0.0 else alpha * unsmoothed + (1 - alpha) * ss.smoothed_level
        display_level = ss.smoothed_level
        hist = ss.rms_history
        hist.append(display_level)
        if len(hist) > 240:
            del hist[: len(hist) - 240]
        ss.max_rms = max(ss.max_rms, display_level)

    if not ss.timer_running and raw_rms > 0.0 and processor_status == 'ready':
        ss.timer_running = True
        ss.started_at = now
        ss.last_update = now

    if ss.debug_override_enabled:
        display_level = float(ss.debug_level_override)

    threshold = float(ss.noise_threshold)
    if not ss.timer_running or display_level <= threshold:
        drain_ml = 0.0
    else:
        frac = (display_level - threshold) / max(1e-6, (1.2 - threshold))
        frac = min(1.0, max(0.0, frac))
        drain_ml = baseline_rate_per_sec * dt * frac

    penalty_ml = 0.0
    now_ts = now
    if ss.timer_running and display_level > threshold and (now_ts - ss.last_penalty_time) > 1.5:
        penalty_fraction = ss.penalty_pct / 100.0
        penalty_ml = ss.water_left_ml * penalty_fraction
        ss.water_left_ml = max(0.0, ss.water_left_ml - penalty_ml)
        ss.last_penalty_time = now_ts
        ss.tilt_until = now_ts + 1.5
        ss.penalty_events += 1
        ss.last_penalty_intensity = max(0.0, display_level - threshold)

    ss.water_left_ml = max(0.0, ss.water_left_ml - drain_ml)

    elapsed = 0.0 if not ss.timer_running or ss.started_at is None else (now - ss.started_at)
    if ss.timer_running:
        if elapsed >= class_secs or ss.water_left_ml <= 0.0:
            ss.done = True

    col1, col2 = st.columns([1,1])
    with col1:
        tilt_active = (ss.tilt_until > now)
        spill_intensity = ss.last_penalty_intensity if tilt_active else 0.0
        render_glass(ss.water_left_ml, ml_total, tilt=tilt_active, spill_intensity=spill_intensity)
        st.markdown(f"<div style='text-align:center; font-size:1.2rem; margin-top:-12px;'>{ss.water_left_ml:.1f} ml</div>", unsafe_allow_html=True)
    with col2:
        if ss.timer_running:
            st.metric("Nåværende volum", f"{display_level:.2f}")
            st.caption("(0≈stille → 1 bråk).")
            st.metric("Høyeste volum", f"{st.session_state.max_rms:.2f}")
            st.caption(f"RMS: {raw_rms:.5f}")
            st.metric("Straffer", f"{ss.penalty_events}")
        else:
            st.info("Venter på lyd for å starte klokka… lag en liten lyd.")
        ss.noise_threshold = st.slider("Støysterskel", 0.5, 1.4, float(ss.noise_threshold), 0.01, help="Over linja tappes glasset. Passeres den får dere også et søl (1.5s cooldown).")
        projected_break = ss.water_left_ml
    st.metric("Gjenværende pause", f"{projected_break:.1f} min")
    remaining = max(0.0, class_secs - elapsed)
    st.caption(f"Tid igjen: {int(remaining//60)}m {int(remaining%60)}s")

    recent_penalty = (now - ss.last_penalty_time) < 1.5
    if recent_penalty and ss.last_penalty_time > ss.last_sound_played_time:
        audio_tag = ""
        if getattr(ss, 'alert_sound_b64', ''):
            audio_tag = f"<audio autoplay style='display:none;'><source src=\"data:audio/mp3;base64,{ss.alert_sound_b64}\" type=\"audio/mpeg\" /></audio>"
        st.markdown(f"{audio_tag}<div style='text-align:center; font-size:2rem; font-weight:700; color:#ff2d2d;'>STFU! 🤫</div>", unsafe_allow_html=True)
        ss.last_sound_played_time = ss.last_penalty_time

    if ss.rms_history:
        try:
            import pandas as pd, altair as alt
            df = pd.DataFrame({'idx': list(range(len(ss.rms_history))), 'level': ss.rms_history})
            thr = float(ss.noise_threshold)
            level_line = alt.Chart(df).mark_line(color="#1f77b4").encode(x='idx', y=alt.Y('level', scale=alt.Scale(domain=[0,1.25])))
            thr_rule = alt.Chart(pd.DataFrame({'thr':[thr]})).mark_rule(color='red').encode(y='thr')
            st.altair_chart((level_line + thr_rule).properties(height=260), use_container_width=True)
        except Exception:
            st.line_chart(ss.rms_history, height=260)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Nullstill glass"):
            ss.water_left_ml = float(ss.initial_water_ml)
    with c2:
        if st.button("Avslutt time"):
            ss.done = True

    with st.expander("Manuell kontroll / juksepanel", expanded=False):
        ss.debug_override_enabled = st.checkbox("Overstyr nivå manuelt", value=ss.debug_override_enabled)
        if ss.debug_override_enabled:
            ss.debug_level_override = st.slider("Testnivå", 0.0, 1.5, float(ss.debug_level_override), 0.01)
            st.caption("Denne verdien erstatter målt nivå (lek litt).")

    if not ss.done:
        st_autorefresh = __import__('streamlit_autorefresh').st_autorefresh
        st_autorefresh(interval=500, key='tick')
    else:
        final_break = min(ss.max_break_minutes, ss.water_left_ml)
        st.success(f"Timen er ferdig. Pause: {final_break:.1f} minutter")
        if st.button("Til oppsett"):
            ss.configured = False
            st.rerun()  # fallback if rerun import path differs
