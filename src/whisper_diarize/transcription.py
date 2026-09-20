"""Speech-to-text transcription using faster-whisper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from whisper_diarize.models import WordItem

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from whisper_diarize.audio import StageProgressFn

_MODEL_ALIASES = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "openai/whisper-large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
    "distil-large-v3": "distil-large-v3",
    "distil-whisper/distil-large-v3": "distil-large-v3",
    "distil-large-v3.5": "distil-large-v3.5",
    "distil-whisper/distil-large-v3.5": "distil-large-v3.5",
}


def resolve_model_id(model_size: str) -> str:
    """Resolve short aliases to concrete model IDs."""
    model = model_size.strip()
    if not model:
        raise ValueError("Model name cannot be empty.")
    return _MODEL_ALIASES.get(model, model)


def is_qwen_asr_model(model_size: str) -> bool:
    """Return True when the model ID should use the Qwen3-ASR backend."""
    lowered = model_size.strip().lower()
    return "qwen3-asr" in lowered


def words_from_transcript(text: str, start_s: float, end_s: float) -> list[WordItem]:
    """Split transcript text into words and spread them across a time span."""
    tokens = text.split()
    if not tokens:
        return []
    if len(tokens) == 1:
        return [WordItem(start_s=start_s, end_s=end_s, word=tokens[0])]
    duration = max(end_s - start_s, 0.0)
    step = duration / len(tokens)
    return [
        WordItem(
            start_s=start_s + (step * i),
            end_s=start_s + (step * (i + 1)),
            word=token,
        )
        for i, token in enumerate(tokens)
    ]


def transcribe(
    audio: NDArray[np.floating],
    sr: int,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8_float16",
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    condition_on_previous_text: bool = False,
    temperature: float | list[float] | tuple[float, ...] | None = None,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 3,
    on_progress: StageProgressFn | None = None,
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
        condition_on_previous_text: Feed previous tokens into each next segment.
            Disabling this can reduce long-form repetition artifacts.
        temperature: Sampling temperature, or fallback schedule when tuple/list.
            When None, faster-whisper uses its built-in fallback schedule.
        repetition_penalty: Penalize repeated token generation (>1.0 discourages loops).
        no_repeat_ngram_size: Prevent repeating n-grams of this size.
        on_progress: Reports (description, local_fraction) within this stage.
            Model loading is ~10% of the cost, segment transcription scales
            linearly with audio duration covering the remaining ~90%.
    """
    resolved_model = resolve_model_id(model_size)
    if is_qwen_asr_model(resolved_model):
        from whisper_diarize.qwen_asr import transcribe as transcribe_qwen

        return transcribe_qwen(
            audio,
            sr,
            model_size=resolved_model,
            device=device,
            language=language,
            on_progress=on_progress,
        )

    from faster_whisper import WhisperModel

    if on_progress:
        on_progress("Loading transcription model", 0.0)
    model = WhisperModel(resolved_model, device=device, compute_type=compute_type)

    if on_progress:
        on_progress("Transcribing speech", 0.10)
    if temperature is None:
        segments, _info = model.transcribe(
            audio,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    else:
        segments, _info = model.transcribe(
            audio,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
        )

    audio_duration = len(audio) / sr
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
        if on_progress and audio_duration > 0:
            frac = 0.10 + 0.90 * min(float(seg.end) / audio_duration, 1.0)
            on_progress("Transcribing speech", frac)

    return words
