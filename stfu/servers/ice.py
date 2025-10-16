import os
import requests
import streamlit as st

# fetch from Metered endpoint 
@st.cache_data(show_spinner=False, ttl=3600)  # cache for 1 hour
def get_ice_servers():
    """
    Get ICE servers from Metered's TURN credentials endpoint.
    Falls back to secrets-based config if the fetch fails or returns nothing.
    """
    api_key = (
        st.secrets.get("METERED_API_KEY")  # preferred: put api key in secrets
        or os.getenv("METERED_API_KEY")    # alt: env var
    )

    if not api_key:
        st.error("No Metered API key configured")
        raise ValueError("No Metered API key configured")
        

    url = "https://stfu.metered.live/api/v1/turn/credentials"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Expect a list like:
        # [{"urls":"stun:..."}, {"urls":"turn:...","username":"...","credential":"..."} ...]
        ice_servers = []
        if isinstance(data, list):
            for s in data:
                if isinstance(s, dict) and "urls" in s:
                    entry = {"urls": s["urls"]}
                    if "username" in s:
                        entry["username"] = s["username"]
                    if "credential" in s:
                        entry["credential"] = s["credential"]
                    ice_servers.append(entry)

        if ice_servers:
            return ice_servers
        else:
            st.warning("No ICE servers returned from Metered")
            return []
    except Exception as e:
        st.error(f"Failed to fetch ICE servers: {e}")
        return []