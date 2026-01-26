"""High-level pipeline for diarized transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf

from sales_agent import audio, diarization, transcription
from sales_agent.alignment import align_words_to_speakers
from sales_agent.models import Utterance
from sales_agent.output import write_all

if TYPE_CHECKING:
    from sales_agent.models import SpeakerTurn, WordItem


@dataclass
class PipelineConfig:
    """Configuration for the transcription pipeline."""

    # Audio cleaning
    clean_audio: bool = True
    highpass_freq: int = 80
    noise_reduction_strength: float = 0.8

    # Transcription
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str | None = None

    # Diarization
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    # Alignment
    max_gap_s: float = 0.9
    gap_tolerance: float = 0.5


@dataclass
class PipelineResult:
    """Result from running the pipeline."""

    utterances: list[Utterance]
    turns: list[SpeakerTurn]
    words: list[WordItem]
    output_paths: dict[str, Path] = field(default_factory=dict)


def run(
    input_path: Path,
    hf_token: str,
    config: PipelineConfig | None = None,
    save_cleaned: bool = False,
) -> PipelineResult:
    """
    Run the full diarized transcription pipeline.

    Args:
        input_path: Path to input audio file
        hf_token: Hugging Face token for pyannote models
        config: Pipeline configuration (uses defaults if None)
        save_cleaned: Save cleaned audio as .cleaned.wav
    """
    if config is None:
        config = PipelineConfig()

    out_base = input_path.with_suffix("")

    # Load audio
    audio_data, sr = audio.load_mono_16k(input_path)

    # Clean audio
    if config.clean_audio:
        audio_data = audio.clean(
            audio_data,
            sr,
            highpass_freq=config.highpass_freq,
            noise_reduction_strength=config.noise_reduction_strength,
        )
        if save_cleaned:
            cleaned_path = out_base.with_suffix(".cleaned.wav")
            sf.write(str(cleaned_path), audio_data, sr)

    # Save temp WAV for diarization
    tmp_wav = out_base.with_suffix(".tmp_16k_mono.wav")
    sf.write(str(tmp_wav), audio_data, sr)

    try:
        # Diarize
        turns = diarization.diarize(
            tmp_wav,
            token=hf_token,
            model_id=config.diarization_model,
            num_speakers=config.num_speakers,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
        )

        # Transcribe
        words = transcription.transcribe(
            audio_data,
            sr,
            model_size=config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
        )

        # Align
        utterances = align_words_to_speakers(
            turns,
            words,
            max_gap_s=config.max_gap_s,
            gap_tolerance=config.gap_tolerance,
        )

        # Write outputs
        output_paths = write_all(out_base, utterances)

    finally:
        tmp_wav.unlink(missing_ok=True)

    return PipelineResult(
        utterances=utterances,
        turns=turns,
        words=words,
        output_paths=output_paths,
    )

