# 🥛 Quiet Break — Copilot Instructions

## Project Overview

This Streamlit project creates an interactive classroom noise monitor. The interface displays a virtual glass of water that drains faster when the room gets noisy, using live microphone input. The remaining water at the end of the class determines the students' break time.

## Key Features

* **Setup page:** Teacher sets class duration, initial water volume, noise sensitivity, and max break time.
* **Live class page:** Monitors microphone input via WebRTC (`streamlit-webrtc`) and drains the glass dynamically.
* **Visualization:** Simple SVG rendering of the glass updates as noise increases.
* **Break calculation:** Remaining water (ml) ≈ break minutes (capped at max break).

## Main Files

* **`app.py`** — Core Streamlit application.

  * Contains two main functions: `settings_page()` and `class_page()`.
  * Uses a `NoiseProcessor` class (inherits `AudioProcessorBase`) to compute RMS noise levels.
  * Uses `streamlit_autorefresh` for smooth, periodic UI updates.
* **`requirements.txt`** *(optional)* — Typical dependencies:

  ```
  streamlit
  streamlit-webrtc
  streamlit-autorefresh
  numpy
  ```

## Development Commands

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
project-root/
│
├── app.py                   # Main Streamlit app
├── copilot-instructions.md  # This file
├── requirements.txt         # Dependencies
└── README.md                # General documentation (optional)
```

## Copilot Tips

* **Navigation logic:** The app uses `st.session_state.configured` to route between setup and class pages. Any new pages should follow the same pattern.
* **State management:** Keep all new session state keys inside `init_state()` to ensure consistent initialization.
* **Mic input:** If modifying noise logic, adjust scaling in `NoiseProcessor.recv_audio()` (the `compressed * 8.0` part controls sensitivity curve).
* **Visual updates:** SVG in `render_glass()` can be replaced with more advanced visuals, but keep the height/width ratio consistent.
* **Auto-refresh:** `st_autorefresh(interval=500, key="tick")` controls UI refresh rate (2x/sec). Tune for performance.
* **Extensibility ideas:**

  * Add pause/resume logic via a `paused` flag in session state.
  * Add a history or log of noise levels.
  * Save break-time results per session.

## Example Additions

* **Persistent session logs:** Store end-of-class results in a local CSV.
* **Classroom leaderboard:** Track quietest classes over time.
* **Alternative UI:** Replace glass with progress bar or animated waves.

## Troubleshooting

* `AttributeError: module 'streamlit' has no attribute 'experimental_rerun'` → Use `st.rerun()` instead.
* `st.autorefresh()` not found → install `streamlit-autorefresh`.
* No mic input → ensure browser permissions allow microphone access.

---

**Author:** Nicolai Olsen
**Frameworks:** Streamlit + streamlit-webrtc
**Purpose:** Lighthearted classroom engagement & noise discipline tool.
