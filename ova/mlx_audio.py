import io
import wave

import mlx.core as mx
import numpy as np


def mx_to_wav_bytes(arr: mx.array, sr: int) -> bytes:
    # Ensure float32 in [-1, 1]
    if arr.dtype == mx.int16:
        arr = arr.astype(mx.float32) / 32768.0
    else:
        arr = arr.astype(mx.float32)
        arr = mx.clip(arr, -1.0, 1.0)

    # ---- RMS normalize (mlx version) ----
    eps = 1e-8
    target_rms = 0.15
    peak_limit = 0.90
    rms = mx.sqrt(mx.mean(arr * arr) + eps)
    arr = arr * (target_rms / mx.maximum(rms, eps))

    # prevent clipping
    peak = mx.max(mx.abs(arr)) + eps
    if peak > peak_limit:
        arr = arr * (peak_limit / peak)

    arr = mx.clip(arr, -1.0, 1.0)

    # Convert to int16
    arr_i16 = (arr * 32767.0).astype(mx.int16)

    # Convert to NumPy for wave module
    arr_i16_np = np.array(arr_i16)

    # Ensure shape = (samples, channels)
    if arr_i16_np.ndim == 1:
        arr_i16_np = arr_i16_np[:, None]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(arr_i16_np.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr_i16_np.tobytes())

    return buf.getvalue()


def rms_normalize(
    arr: np.ndarray, target_rms=0.15, peak_limit=0.90, eps=1e-12
) -> np.ndarray:
    x = arr.astype(np.float32)

    rms = np.sqrt(np.mean(x * x) + eps)
    if rms < eps:
        return x  # silence

    x = x * (target_rms / rms)

    # prevent clipping
    peak = np.max(np.abs(x)) + eps
    if peak > peak_limit:
        x = x * (peak_limit / peak)

    return x
