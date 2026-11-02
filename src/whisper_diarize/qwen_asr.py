"""Batched Qwen3-ASR transcription with measured word timestamps."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TYPE_CHECKING, Any

from whisper_diarize.models import WordItem

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from whisper_diarize.audio import StageProgressFn

_CHUNK_S = 20.0
_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
_MODEL_LOCK = RLock()


@lru_cache(maxsize=4)
def _load(model_id: str, device: str, aligner: bool = False) -> tuple[Any, Any]:
    import torch

    # Direct imports avoid Transformers' lazy export race during parallel cold starts.
    from transformers.models.auto.modeling_auto import (
        AutoModelForMultimodalLM,
        AutoModelForTokenClassification,
    )
    from transformers.models.auto.processing_auto import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model_class = AutoModelForTokenClassification if aligner else AutoModelForMultimodalLM
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model_class.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return processor, model


def _transcribe_batch(
    paths: list[str], model_id: str, device: str, language: str | None, duration_s: float
) -> list[str]:
    import torch

    # Directory workers share cached models; generation must not overlap on one model.
    with _MODEL_LOCK, torch.inference_mode():
        processor, model = _load(model_id, device)
        inputs = processor.apply_transcription_request(
            audio=paths,
            language=language or "English",
        ).to(model.device, model.dtype)
        max_new_tokens = min(1024, max(64, int(duration_s * 20) + 32))
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return [
            text.strip()
            for text in processor.decode(generated_ids, return_format="transcription_only")
        ]


def transcribe_array(
    audio: NDArray[np.floating],
    sr: int,
    model_id: str = "Qwen/Qwen3-ASR-1.7B-hf",
    device: str = "cuda",
    language: str | None = "en",
) -> str:
    """Transcribe a single audio array and return plain text."""
    import soundfile as sf

    if len(audio) == 0:
        return ""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "audio.wav"
        sf.write(path, audio, sr)
        return _transcribe_batch([str(path)], model_id, device, language, len(audio) / sr)[0]


def transcribe(
    audio: NDArray[np.floating],
    sr: int,
    model_size: str = "Qwen/Qwen3-ASR-1.7B-hf",
    device: str = "cuda",
    language: str | None = "en",
    on_progress: StageProgressFn | None = None,
    **_kwargs: object,
) -> list[WordItem]:
    """Transcribe 20-second chunks in batches and force-align each transcript."""
    import soundfile as sf
    import torch

    if len(audio) == 0:
        return []
    batch_size = 4 if device == "cuda" else 1
    chunk_samples = int(_CHUNK_S * sr)
    words: list[WordItem] = []
    if on_progress:
        on_progress("Loading transcription models", 0.0)

    with TemporaryDirectory() as tmp:
        for batch_start in range(0, len(audio), chunk_samples * batch_size):
            offsets = list(
                range(
                    batch_start,
                    min(batch_start + chunk_samples * batch_size, len(audio)),
                    chunk_samples,
                )
            )
            paths = []
            for i, offset in enumerate(offsets):
                path = Path(tmp) / f"chunk-{i}.wav"
                sf.write(path, audio[offset : offset + chunk_samples], sr)
                paths.append(str(path))
            duration = min(chunk_samples, len(audio) - batch_start) / sr
            texts = _transcribe_batch(paths, model_size, device, language, duration)
            for path, text, offset in zip(paths, texts, offsets, strict=True):
                if not text:
                    continue
                with _MODEL_LOCK, torch.inference_mode():
                    processor, model = _load(_ALIGNER_ID, device, aligner=True)
                    inputs, word_lists = processor.prepare_forced_aligner_inputs(
                        audio=path,
                        transcript=text,
                        language=language or "English",
                    )
                    inputs = inputs.to(model.device, model.dtype)
                    output = model(**inputs)
                    timestamps = processor.decode_forced_alignment(
                        logits=output.logits,
                        input_ids=inputs["input_ids"],
                        word_lists=word_lists,
                        timestamp_token_id=model.config.timestamp_token_id,
                    )[0]
                chunk_end = min(offset + chunk_samples, len(audio)) / sr
                for item in timestamps:
                    start = min(chunk_end, max(offset / sr, offset / sr + item["start_time"]))
                    end = min(chunk_end, max(start, offset / sr + item["end_time"]))
                    words.append(WordItem(start_s=start, end_s=end, word=item["text"]))
            if on_progress:
                completed = min(batch_start + chunk_samples * batch_size, len(audio))
                on_progress("Transcribing and aligning words", completed / len(audio))
    return words
