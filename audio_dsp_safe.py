"""Boundary-safe overlap-add and envelope smoothing, preserving sample count."""
import numpy as np


def smooth_edge(x, width):
    width = max(1, int(width))
    if not len(x):
        return x.copy()
    left, right = width // 2, width - 1 - width // 2
    return np.convolve(np.pad(x, (left, right), mode="edge"), np.ones(width) / width, mode="valid")


def spectral_gain(y, gain, frame=4096, hop=1024):
    """Centered, padded STFT. Unity gain round trips edges and fractional last hops."""
    y = np.asarray(y, dtype=np.float32)
    out = y.copy()
    if not len(y):
        return out
    win = np.hanning(frame).astype(np.float64)
    pad = frame
    for ch in range(y.shape[1]):
        x = np.pad(y[:, ch], (pad, pad + (-len(y) % hop)), mode="edge").astype(np.float64)
        acc, norm = np.zeros(len(x)), np.zeros(len(x))
        for a in range(0, len(x) - frame + 1, hop):
            spec = np.fft.rfft(x[a:a+frame] * win)
            rec = np.fft.irfft(spec * gain, n=frame)
            acc[a:a+frame] += rec * win
            norm[a:a+frame] += win ** 2
        out[:, ch] = (acc / np.maximum(norm, 1e-12))[pad:pad+len(y)]
    return out
