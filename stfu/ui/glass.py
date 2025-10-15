import streamlit as st


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