"""Speech-to-text transcription using faster-whisper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sales_agent.models import WordItem

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def transcribe(
    audio: NDArray[np.floating],
    sr: int,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8_float16",
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> list[WordItem]:
    """
    Transcribe audio to words with timestamps.

    Args:
        audio: Audio array (16kHz mono)
        sr: Sample rate
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
        device: Device to use (cuda, cpu)
        compute_type: CTranslate2 compute type (int8_float16, float16, int8, float32)
        language: Language code (None for auto-detect)
        beam_size: Beam search size
        vad_filter: Use voice activity detection filter
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        audio,
        language=language,
        vad_filter=vad_filter,
        word_timestamps=True,
        beam_size=beam_size,
    )

    words: list[WordItem] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            words.append(
                WordItem(
                    start_s=float(w.start),
                    end_s=float(w.end),
                    word=w.word,
                )
            )

    return words

