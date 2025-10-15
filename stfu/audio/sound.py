import os, base64

def load_sound_b64(filename: str) -> str:
    try:
        path = os.path.join(os.path.dirname(__file__), filename)
        path = os.path.abspath(path)
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('ascii')
    except Exception as e:
        print(f"Failed loading sound {filename}: {e}")
        return ""
