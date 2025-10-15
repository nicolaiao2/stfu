import math
import numpy as np
from streamlit_webrtc import AudioProcessorBase

class NoiseProcessor(AudioProcessorBase):
    """Compute a simple RMS-based mic loudness indicator."""
    def __init__(self) -> None:
        self.level = 0.0
        self.raw_rms = 0.0
        self._frame_count = 0
        self._logged_shapes = 0

    def recv(self, frame):  # type: ignore
        audio = frame.to_ndarray()
        if audio.ndim == 2:
            if audio.shape[0] <= 8 and audio.shape[0] <= audio.shape[1]:
                audio = audio.mean(axis=0)
            elif audio.shape[1] <= 8 and audio.shape[1] <= audio.shape[0]:
                audio = audio.mean(axis=1)
            else:
                audio = audio.mean(axis=0)
        x = audio.astype(np.float32)
        if x.size > 0 and np.max(np.abs(x)) > 2.0:
            x = x / 32768.0
        x = np.clip(x, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        self.raw_rms = rms
        compressed = math.pow(rms, 0.5)
        self.level = min(1.5, compressed * 4.0)
        if self._logged_shapes < 3:
            first_samples = x[:5]
            print(
                f"Frame {self._frame_count} len={len(x)} first5={np.array2string(first_samples, precision=4)} rms={rms:.6f} level={self.level:.4f}"
            )
            self._logged_shapes += 1
        self._frame_count += 1
        return frame
