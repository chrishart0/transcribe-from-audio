"""Speaker diarization with Nemotron 3's eight-speaker model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from whisper_diarize.audio import load_mono_16k
from whisper_diarize.models import SpeakerTurn

if TYPE_CHECKING:
    from whisper_diarize.audio import StageProgressFn

MODEL_ID = "nvidia/Nemotron-3-Diarization"


def diarize(
    wav_path: Path, device: str = "cuda", on_progress: StageProgressFn | None = None
) -> list[SpeakerTurn]:
    """Keep short and overlapping turns for word-level speaker assignment."""
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForAudioFrameClassification
    from transformers.models.auto.processing_auto import AutoProcessor

    if on_progress:
        on_progress("Loading Nemotron diarization model", 0.0)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID).to(device).eval()
    audio, sr = load_mono_16k(wav_path)
    if len(audio) == 0:
        return []
    inputs = processor(audio, sampling_rate=sr).to(device, dtype=model.dtype)
    if on_progress:
        on_progress("Diarizing speakers", 0.15)
    with torch.inference_mode():
        logits = model(**inputs).logits
    segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
    turns = [
        SpeakerTurn(
            start_s=float(s["Start"]),
            end_s=float(s["End"]),
            speaker=f"SPEAKER_{int(s['Speaker']):02}",
        )
        for s in segments
    ]
    turns.sort(key=lambda t: t.start_s)
    if on_progress:
        on_progress("Diarizing speakers", 1.0)
    return turns
