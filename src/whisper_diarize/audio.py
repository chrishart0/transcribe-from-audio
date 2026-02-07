"""Audio loading and preprocessing."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Callback for reporting progress: (description, local_fraction 0.0-1.0)
StageProgressFn = Callable[[str, float], None]

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"})
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def find_media_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Find supported audio/video files in a directory.

    Returns sorted list of paths with supported extensions.
    """
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort()
    return files


def is_video(path: Path) -> bool:
    """Check if a file is a video based on its extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def extract_audio(input_path: Path) -> Path:
    """Extract audio from a video file using ffmpeg, returning path to a temp WAV."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed to extract audio: {result.stderr}")

    return tmp_path


def load_mono_16k(input_path: Path) -> tuple[NDArray[np.floating], int]:
    """Load audio file as mono 16kHz numpy array."""
    audio, sr = librosa.load(str(input_path), sr=16000, mono=True)
    return audio, int(sr)


def clean(
    audio: NDArray[np.floating],
    sr: int,
    reduce_noise: bool = True,
    highpass_freq: int = 80,
    normalize: bool = True,
    noise_reduction_strength: float = 0.8,
    on_progress: StageProgressFn | None = None,
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
        on_progress: Reports (description, local_fraction) within this stage.
            Noise reduction dominates cost (~80%), filtering and normalization
            are near-instant.
    """
    import noisereduce as nr
    from scipy.signal import butter, sosfilt

    cleaned = audio.copy()

    if highpass_freq > 0:
        if on_progress:
            on_progress("Filtering audio", 0.0)
        sos = butter(5, highpass_freq, btype="high", fs=sr, output="sos")
        filtered = sosfilt(sos, cleaned)
        cleaned = np.asarray(filtered, dtype=np.float32)

    if reduce_noise:
        if on_progress:
            on_progress("Reducing noise", 0.05)
        cleaned = nr.reduce_noise(
            y=cleaned,
            sr=sr,
            prop_decrease=noise_reduction_strength,
            stationary=False,
        )

    if normalize:
        if on_progress:
            on_progress("Normalizing audio", 0.95)
        peak = np.abs(cleaned).max()
        if peak > 0:
            cleaned = cleaned / peak * 0.95

    if on_progress:
        on_progress("Cleaning audio", 1.0)

    return cleaned
