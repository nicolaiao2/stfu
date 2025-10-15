import time
import math
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Quiet Break Timer", page_icon="🥛", layout="centered")

# ------------------------------
# Session-state helpers
# ------------------------------

def init_state():
    ss = st.session_state
    ss.setdefault("configured", False)
    ss.setdefault("class_minutes", 45)
    ss.setdefault("initial_water_ml", 45.0)  # 1 ml == 1 break minute mapping (kept for backward compatibility)
    ss.setdefault("max_break_minutes", 15.0)
    ss.setdefault("started_at", None)
    ss.setdefault("last_update", None)
    ss.setdefault("water_left_ml", ss["initial_water_ml"]) 
    ss.setdefault("done", False)
    # Diagnostics
    ss.setdefault("max_rms", 0.0)
    ss.setdefault("rms_history", [])  # list of floats (post-compression/display level)
    ss.setdefault("noise_floor_rms", 0.0)  # dynamic floor estimate (raw RMS domain)
    ss.setdefault("smoothed_level", 0.0)   # smoothed post-floor, post-compression level (0-1.2)
    ss.setdefault("smoothing_alpha", 0.3)   # smoothing factor (0-1)
    # Debug overrides
    ss.setdefault("debug_override_enabled", False)
    ss.setdefault("debug_level_override", 0.0)
    # Loud penalty feature
    ss.setdefault("penalty_pct", 5.0)        # % of current remaining water to spill on loud event
    ss.setdefault("tilt_until", 0.0)         # timestamp until which glass is tilted
    ss.setdefault("last_penalty_time", 0.0)  # last time (epoch) penalty applied
    ss.setdefault("penalty_events", 0)       # count of loud penalties triggered
    ss.setdefault("last_penalty_intensity", 0.0)  # display_level - threshold at last penalty
    ss.setdefault("last_sound_played_time", 0.0)  # prevent repeated audio spam
    ss.setdefault("noise_threshold", 1.0)   # threshold level above which draining occurs
    # Load custom alert sound once
    if "alert_sound_b64" not in ss:
        try:
            import base64, os
            sound_path = os.path.join(os.path.dirname(__file__), "hey-42237.mp3")
            with open(sound_path, "rb") as f:
                ss.alert_sound_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            print(f"Failed to load alert sound: {e}")
            ss.alert_sound_b64 = ""
    # Timer control: start only after audio actually flowing
    ss.setdefault("timer_running", False)
    # removed pending_timer_start (auto start on first audio)

init_state()

# ------------------------------
# Visual: simple SVG glass
# ------------------------------

def render_glass(ml_left: float, ml_total: float, tilt: bool = False, tilt_deg: float = -25.0, spill_intensity: float = 0.0):
        pct = 0.0 if ml_total <= 0 else max(0.0, min(1.0, ml_left / ml_total))
        # Add padding around SVG to avoid clipping when rotated
        width, height = 220, 360
        inner_offset_x = 20
        glass_width = 180
        glass_height = 320
        glass_x = inner_offset_x + 40
        glass_y = 12
        inner_water_height = glass_height - 24
        water_h = int(pct * inner_water_height)
        water_bottom_y = glass_y + inner_water_height
        water_y = water_bottom_y - water_h
        rotate_attr = f"transform='rotate({tilt_deg} {width/2} {height/2})'" if tilt else ""
        # Coordinates for glass rectangle (upright)
        rect_width = glass_width - 80
        rect_height = glass_height - 24
        rim_right_x = glass_x + rect_width  # top-right rim point when upright
        rim_right_y = glass_y
        spill_path = ""
        water_color = "#7ec8e3"
        stroke_color = "#555"
        # If tilting (penalty), shift colors toward red for emphasis
        if tilt:
            water_color = "#ff6b6b"
            stroke_color = "#c0392b"
        if tilt and pct > 0.02:
            # Dynamic scaling: map spill_intensity (0.. ~0.8) to length/offset factors
            intensity = max(0.0, spill_intensity)
            norm_int = min(1.0, intensity / 0.8)  # clamp
            dx_far = 40 + 50 * norm_int   # horizontal reach of the main curve
            dy_drop = 18 + 22 * norm_int  # vertical downward extent
            droplet_r = 5 + 3 * norm_int
            mid_out = 20 + 25 * norm_int
            # Main spill path (stroke) – bezier then curve downward
            spill_path = (
                f"<path d='M {rim_right_x} {rim_right_y} c {dx_far*0.45:.1f} {-6 - 6*norm_int:.1f} {dx_far*0.75:.1f} {6 + 8*norm_int:.1f} {dx_far:.1f} {mid_out:.1f} "
                f"q {-15 - 10*norm_int:.1f} {-5 - 4*norm_int:.1f} {-30 - 12*norm_int:.1f} {4 + 4*norm_int:.1f} q {8 + 4*norm_int:.1f} {5 + 3*norm_int:.1f} {14 + 6*norm_int:.1f} {14 + 6*norm_int:.1f} "
                f"q {-18 - 10*norm_int:.1f} {-8 - 4*norm_int:.1f} {-38 - 10*norm_int:.1f} {-6 - 4*norm_int:.1f}"
                f"' fill='none' stroke='{water_color}' stroke-width='{4 + 2*norm_int:.1f}' stroke-linecap='round' stroke-opacity='{0.75 + 0.2*norm_int:.2f}'>"
                f"<animate attributeName='opacity' values='{0.9 + 0.1*norm_int:.2f};0' dur='1.2s' repeatCount='1' />"
                f"</path>"
                f"<circle cx='{rim_right_x + dx_far:.1f}' cy='{rim_right_y + mid_out + dy_drop*0.3:.1f}' r='{droplet_r:.1f}' fill='{water_color}' fill-opacity='{0.6 + 0.3*norm_int:.2f}'>"
                f"<animate attributeName='cy' from='{rim_right_y + mid_out + dy_drop*0.3:.1f}' to='{rim_right_y + mid_out + dy_drop + 30:.1f}' dur='1.2s' repeatCount='1' />"
                f"<animate attributeName='opacity' values='1;0' dur='1.2s' repeatCount='1' />"
                f"</circle>"
            )

        svg = f"""
            <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
                <g {rotate_attr}>
                    <rect x="{glass_x}" y="{glass_y}" width="{rect_width}" height="{rect_height}"
                        rx="16" ry="16" fill="none" stroke="{stroke_color}" stroke-width="4"/>
                    <rect x="{glass_x+2}" y="{water_y}" width="{rect_width-4}" height="{water_h}"
                        rx="12" ry="12" fill="{water_color}" />
                        {spill_path}
            </svg>
        """
        st.markdown(svg, unsafe_allow_html=True)
        
# ------------------------------
# Audio processing
# ------------------------------

class NoiseProcessor(AudioProcessorBase):
    """Compute a simple RMS-based mic loudness indicator.

    The value is accessible as `self.level` in ~[0, 1.5].
    """
    def __init__(self) -> None:
        print("NoiseProcessor.__init__ invoked — audio processor constructed")
        self.level = 0.0
        self.raw_rms = 0.0
        self._frame_count = 0
        self._logged_shapes = 0

    # NOTE: For streamlit-webrtc AudioProcessorBase, the correct callback is `recv`.
    # Previously named recv_audio, which meant frames were never processed.
    def recv(self, frame):
        audio = frame.to_ndarray()
        # Normalize shape to 1-D samples. PyAV typically returns (channels, samples).
        if audio.ndim == 2:
            # Heuristic: interpret smaller dimension as channels.
            if audio.shape[0] <= 8 and audio.shape[0] <= audio.shape[1]:  # (channels, samples)
                audio = audio.mean(axis=0)
            elif audio.shape[1] <= 8 and audio.shape[1] <= audio.shape[0]:  # (samples, channels)
                audio = audio.mean(axis=1)
            else:  # fallback: assume channels x samples
                audio = audio.mean(axis=0)
        x = audio.astype(np.float32)
        # Detect unnormalized int16 delivered as float32 (large magnitudes > 2)
        if x.size > 0 and np.max(np.abs(x)) > 2.0:
            x = x / 32768.0  # assume originally int16 range
        # Hard clip just in case
        x = np.clip(x, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        self.raw_rms = rms
        # Compress curve so low noise drains very little, high noise drains more.
        # Apply a softer compression & scaling to map typical speech (~0.02-0.1 RMS) into ~0-1 range.
        compressed = math.pow(rms, 0.5)  # sqrt curve
        self.level = min(1.5, compressed * 4.0)
        if self._logged_shapes < 5:  # only log first few frames to avoid noise
            first_samples = x[:5]
            print(
                f"Audio frame {self._frame_count} shape(original)={frame.to_ndarray().shape} "
                f"processed_len={len(x)} dtype={x.dtype} first5={np.array2string(first_samples, precision=4)}\n"
                f"RMS: {rms:.6f} compressed:{compressed:.6f} level:{self.level:.6f}"
            )
            self._logged_shapes += 1
        self._frame_count += 1
        return frame

# ------------------------------
# Pages
# ------------------------------

def settings_page():
    st.title("🥛 STFU - Stille Tid For Undervisning")
    st.caption("Sett varighet, maks pause og straff. Klar? Trykk start.")

    st.session_state.class_minutes = st.number_input(
        "Undervisningstid (minutter)", min_value=1, max_value=240, value=int(st.session_state.class_minutes)
    )
    st.session_state.max_break_minutes = float(
        st.number_input("Maks pausetid (min) / startvolum (ml)", min_value=1, max_value=1000, value=int(st.session_state.max_break_minutes))
    )
    # Keep water volume exactly equal to max break minutes (1 ml = 1 break minute)
    st.session_state.initial_water_ml = st.session_state.max_break_minutes
    # (Noise threshold & penalties configured during class; sensitivity removed.)
    st.session_state.penalty_pct = float(
        st.slider("Høylytt straff (% av gjenværende vann som søles)", 1.0, 30.0, float(st.session_state.penalty_pct), 1.0)
    )

    st.write("---")

    if st.button("Start timen", type="primary"):
        ss = st.session_state
        ss.configured = True
        # Defer actual timer start until audio frames are received in class_page
        ss.started_at = None
        ss.last_update = None
        ss.timer_running = False
        ss.water_left_ml = float(ss.initial_water_ml)  # equals max_break_minutes
        ss.done = False
        st.rerun()


def class_page():
    st.title("🥛 Stillepause – I timen")
    st.caption("Blir det bråk lekker glasset! Vann igjen = pausetid.")
    ss = st.session_state

    # Start mic capture; SENDONLY means we only send audio from browser to Python.
    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        # async_processing can delay availability of the processor; disable for simpler debug.
        async_processing=False,
        audio_processor_factory=NoiseProcessor,
        # Provide a STUN server so peers behind NAT can connect; expands chance audio frames arrive.
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]
        },
    )

    ss = st.session_state
    ml_total = float(ss.initial_water_ml)
    class_secs = float(ss.class_minutes) * 60.0

    # Base drain rate so glass empties exactly over class time if perfectly quiet.
    baseline_rate_per_sec = ml_total / class_secs if class_secs > 0 else 0.0

    # Compute time step (only once timer running) and drain accordingly.
    now = time.time()
    if ss.timer_running:
        dt = 0.0 if ss.last_update is None else max(0.0, now - ss.last_update)
        ss.last_update = now
    else:
        dt = 0.0

    # Current noise levels
    display_level = 0.0  # smoothed, post-floor, post-compression
    raw_rms = 0.0        # raw (normalized) RMS from processor
    processor_status = "not-created"
    manual_frames_used = False
    if ctx is not None:
        if getattr(ctx.state, "playing", False):
            if ctx.audio_processor is not None:
                processor_status = "ready"
                raw_rms = float(getattr(ctx.audio_processor, "raw_rms", 0.0))
            else:
                # Try manual frame pull as fallback (helps diagnose why processor not created)
                processor_status = "missing (fallback active)"
                receiver = getattr(ctx, "audio_receiver", None)
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
                            print(f"Manual fallback raw RMS={raw_rms:.5f}")
                    except Exception as e:
                        print(f"Manual frame pull failed: {e}")
        else:
            processor_status = "connecting"

    if processor_status != "ready":
        info_msg = f"Lydstatus: {processor_status}. Tillat mikrofon og vent litt."
        if manual_frames_used:
            info_msg += " (midlertidig reserve aktiv)."
        st.info(info_msg)

    # Dynamic noise floor update & level computation
    if processor_status in ("ready", "missing (fallback active)"):
        nf = ss.noise_floor_rms
        if nf == 0.0:
            ss.noise_floor_rms = raw_rms
        else:
            # If current raw is below (nf * 1.5) treat as near-noise and allow floor to track down.
            decay_alpha = 0.05 if raw_rms < nf * 1.5 else 0.001  # adapt slowly upward, faster downward
            ss.noise_floor_rms = (1 - decay_alpha) * nf + decay_alpha * raw_rms

        effective = max(0.0, raw_rms - ss.noise_floor_rms * 1.2)  # subtract margin (20% over floor)
        # Compress and scale (smaller multiplier now)
        compressed = math.pow(effective, 0.5)
        unsmoothed_level = min(1.2, compressed * 3.0)

        # Exponential smoothing
        alpha = float(ss.smoothing_alpha)
        if ss.smoothed_level == 0.0:
            ss.smoothed_level = unsmoothed_level
        else:
            ss.smoothed_level = alpha * unsmoothed_level + (1 - alpha) * ss.smoothed_level
        display_level = ss.smoothed_level

        # History (store display level)
        hist = ss.rms_history
        hist.append(display_level)
        if len(hist) > 240:
            del hist[: len(hist) - 240]
        ss.max_rms = max(ss.max_rms, display_level)

    # If timer armed but not yet running, start when first meaningful frame arrives (raw_rms > 0)
    if not ss.timer_running and raw_rms > 0.0 and processor_status == "ready":
        ss.timer_running = True
        ss.started_at = now
        ss.last_update = now

    # Drain only when above threshold and timer is running.
    if ss.debug_override_enabled:
        display_level = float(ss.debug_level_override)
    threshold = float(ss.noise_threshold)
    # Drain only if level exceeds threshold and timer running
    if not ss.timer_running or display_level <= threshold:
        drain_ml = 0.0
    else:
        # Scale drain proportionally to how far above threshold (normalized to max ~1.2)
        frac = (display_level - threshold) / max(1e-6, (1.2 - threshold))
        frac = min(1.0, max(0.0, frac))
        drain_ml = baseline_rate_per_sec * dt * frac

    # Penalty event: trigger when smoothed level crosses above teacher-set threshold (only after timer start).
    penalty_ml = 0.0
    now_ts = now
    if ss.timer_running and display_level > threshold and (now_ts - ss.last_penalty_time) > 1.5:
        penalty_fraction = ss.penalty_pct / 100.0
        penalty_ml = ss.water_left_ml * penalty_fraction
        ss.water_left_ml = max(0.0, ss.water_left_ml - penalty_ml)
        ss.last_penalty_time = now_ts
        ss.tilt_until = now_ts + 1.5  # tilt for 1.5s
        ss.penalty_events += 1
        ss.last_penalty_intensity = max(0.0, display_level - threshold)
        print(f"PENALTY: level={display_level:.3f} spilled={penalty_ml:.2f}ml threshold={threshold:.2f} remaining={ss.water_left_ml:.2f}ml")

    print(f"Draining {drain_ml:.3f} ml + penalty {penalty_ml:.3f} (dt={dt:.3f}s, running={ss.timer_running}, raw_rms={raw_rms:.5f}, floor={ss.noise_floor_rms:.5f}, level={display_level:.3f}, threshold={threshold:.2f})")
    ss.water_left_ml = max(0.0, ss.water_left_ml - drain_ml)

    # End conditions
    elapsed = 0.0 if not ss.timer_running or ss.started_at is None else (now - ss.started_at)
    if ss.timer_running:
        if elapsed >= class_secs:
            ss.done = True
        if ss.water_left_ml <= 0.0:
            ss.done = True

    # UI layout
    col1, col2 = st.columns([1, 1])
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
            st.caption(f"Rå RMS: {raw_rms:.5f}")
            st.metric("Straffer", f"{ss.penalty_events}")
        else:
            st.info("Venter på lyd for å starte klokka… lag en liten lyd.")
        ss.noise_threshold = st.slider(
            "Støysterskel",
            0.5, 1.4,
            float(ss.noise_threshold), 0.01,
            help="Over linja tappes glasset. Passeres den får dere også et søl (cooldown 1.5s)."
        )
        # Direct mapping: remaining ml == break minutes (no cap needed because initial equals max)
        projected_break = ss.water_left_ml
    st.metric("Gjenværende pause", f"{projected_break:.1f} min")
    remaining = max(0.0, class_secs - elapsed)
    st.caption(f"Tid igjen: {int(remaining//60)}m {int(remaining%60)}s")

    # Penalty alert (sound + message) - only on fresh penalty
    recent_penalty = (now - ss.last_penalty_time) < 1.5
    if recent_penalty and ss.last_penalty_time > ss.last_sound_played_time:
        audio_tag = ""
        if getattr(ss, "alert_sound_b64", ""):
            audio_tag = f"<audio autoplay style='display:none;'><source src=\"data:audio/mp3;base64,{ss.alert_sound_b64}\" type=\"audio/mpeg\" /></audio>"
        st.markdown(
            f"""
            {audio_tag}
            <div style='text-align:center; font-size:2rem; font-weight:700; color:#ff2d2d;'>STFU! 🤫</div>
            """,
            unsafe_allow_html=True,
        )
        ss.last_sound_played_time = ss.last_penalty_time

    # Waveform + threshold rule
    if st.session_state.rms_history:
        try:
            import pandas as pd, altair as alt
            df = pd.DataFrame({"idx": list(range(len(st.session_state.rms_history))), "level": st.session_state.rms_history})
            thr = float(st.session_state.noise_threshold)
            level_line = alt.Chart(df).mark_line(color="#1f77b4").encode(x="idx", y=alt.Y("level", scale=alt.Scale(domain=[0,1.25])))
            thr_rule = alt.Chart(pd.DataFrame({"thr":[thr]})).mark_rule(color="red").encode(y="thr")
            st.altair_chart((level_line + thr_rule).properties(height=260), use_container_width=True)
        except Exception:
            st.line_chart(st.session_state.rms_history, height=260)

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

    # Auto-refresh the page ~2x per second for live updates
    # (keeps code simple without a manual while-loop)
    if not ss.done:
        st_autorefresh(interval=500, key="tick")
    else:
        final_break = min(ss.max_break_minutes, ss.water_left_ml)
        st.success(f"Timen er ferdig. Pause: {final_break:.1f} minutter")
        if st.button("Til oppsett"):
            ss.configured = False
            st.rerun()


# ------------------------------
# Router
# ------------------------------
if not st.session_state.configured:
    settings_page()
else:
    class_page()
