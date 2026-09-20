"""Qwen3-ASR backend for English-first transcription."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from whisper_diarize.models import WordItem
from whisper_diarize.transcription import words_from_transcript

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from whisper_diarize.audio import StageProgressFn

_CHUNK_S = 20.0
_LANGUAGE_NAMES = {
    "en": "English",
    "english": "English",
}

_processor = None
_model = None
_loaded_model_id: str | None = None


def _language_name(language: str | None) -> str:
    if not language:
        return "English"
    return _LANGUAGE_NAMES.get(language.strip().lower(), language)


def _load(model_id: str, device: str) -> tuple[Any, Any]:
    global _processor, _model, _loaded_model_id
    if _model is not None and _processor is not None and _loaded_model_id == model_id:
        return _processor, _model

    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR requires transformers>=5.13. Install with: uv add 'transformers>=5.13.0'"
        ) from exc

    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype)
    if device == "cuda":
        model = model.to("cuda")
    model.eval()
    _processor = processor
    _model = model
    _loaded_model_id = model_id
    return processor, model


def transcribe_array(
    audio: NDArray[np.floating],
    sr: int,
    model_id: str = "Qwen/Qwen3-ASR-1.7B-hf",
    device: str = "cuda",
    language: str | None = "en",
) -> str:
    """Transcribe a single audio array and return plain text."""
    from tempfile import NamedTemporaryFile

    import soundfile as sf
    import torch

    processor, model = _load(model_id, device)
    language_name = _language_name(language)
    with NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sr)
        inputs = processor.apply_transcription_request(
            audio=tmp.name,
            language=language_name,
        )
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        inputs = inputs.to(model_device, model_dtype)
        max_new_tokens = min(1024, max(64, int((len(audio) / sr) * 20) + 32))
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        text = processor.decode(generated_ids, return_format="transcription_only")[0]
    return (text or "").strip()


def transcribe(
    audio: NDArray[np.floating],
    sr: int,
    model_size: str = "Qwen/Qwen3-ASR-1.7B-hf",
    device: str = "cuda",
    language: str | None = "en",
    on_progress: StageProgressFn | None = None,
    **_kwargs: object,
) -> list[WordItem]:
    """Transcribe long audio with Qwen3-ASR by chunking."""
    if on_progress:
        on_progress("Loading transcription model", 0.0)
    _load(model_size, device)
    if on_progress:
        on_progress("Transcribing speech", 0.10)

    chunk_samples = int(_CHUNK_S * sr)
    total = len(audio)
    if total == 0:
        return []
    if total <= chunk_samples:
        text = transcribe_array(audio, sr, model_id=model_size, device=device, language=language)
        return words_from_transcript(text, 0.0, total / sr)

    words: list[WordItem] = []
    offset = 0
    chunk_index = 0
    n_chunks = max(1, (total + chunk_samples - 1) // chunk_samples)
    while offset < total:
        end = min(offset + chunk_samples, total)
        chunk = audio[offset:end]
        start_s = offset / sr
        end_s = end / sr
        text = transcribe_array(chunk, sr, model_id=model_size, device=device, language=language)
        words.extend(words_from_transcript(text, start_s, end_s))
        chunk_index += 1
        if on_progress:
            frac = 0.10 + 0.90 * min(chunk_index / n_chunks, 1.0)
            on_progress("Transcribing speech", frac)
        offset = end
    return words
