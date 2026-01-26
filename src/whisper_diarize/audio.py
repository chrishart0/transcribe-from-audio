"""Audio loading and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_mono_16k(input_path: Path) -> tuple[NDArray[np.floating], int]:
    """Load audio file as mono 16kHz numpy array."""
    audio, sr = librosa.load(str(input_path), sr=16000, mono=True)
    return audio, sr


def clean(
    audio: NDArray[np.floating],
    sr: int,
    reduce_noise: bool = True,
    highpass_freq: int = 80,
    normalize: bool = True,
    noise_reduction_strength: float = 0.8,
) -> NDArray[np.floating]:
    """
    Clean up muffled/noisy audio.

    Args:
        audio: Input audio array
        sr: Sample rate
        reduce_noise: Apply noise reduction
        highpass_freq: High-pass filter cutoff (0 to disable)
        normalize: Normalize audio levels
        noise_reduction_strength: How aggressive noise reduction is (0-1)
    """
    import noisereduce as nr
    from scipy.signal import butter, sosfilt

    cleaned = audio.copy()

    if highpass_freq > 0:
        sos = butter(5, highpass_freq, btype="high", fs=sr, output="sos")
        cleaned = sosfilt(sos, cleaned).astype(np.float32)

    if reduce_noise:
        cleaned = nr.reduce_noise(
            y=cleaned,
            sr=sr,
            prop_decrease=noise_reduction_strength,
            stationary=False,
        )

    if normalize:
        peak = np.abs(cleaned).max()
        if peak > 0:
            cleaned = cleaned / peak * 0.95

    return cleaned

