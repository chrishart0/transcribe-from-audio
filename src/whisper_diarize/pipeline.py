"""High-level pipeline for diarized transcription."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf

from whisper_diarize import audio, diarization, transcription
from whisper_diarize.alignment import align_words_to_speakers
from whisper_diarize.models import Utterance
from whisper_diarize.output import write_all

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from whisper_diarize.models import SpeakerTurn, WordItem

# Overall progress callback: (stage_description, overall_fraction 0.0-1.0)
ProgressCallback = Callable[[str, float], None]

# Fixed-cost weights (independent of audio length).
# Units are arbitrary — only the ratio to duration-scaled weights matters.
_WEIGHT_EXTRACT_VIDEO = 3.0
_WEIGHT_LOAD_AUDIO = 2.0
_WEIGHT_LOAD_DIARIZE_MODEL = 5.0
_WEIGHT_LOAD_TRANSCRIBE_MODEL = 5.0
_WEIGHT_ALIGN = 1.0
_WEIGHT_WRITE = 1.0

# Duration-scaled weights (cost per minute of audio).
# Noise reduction, diarization inference, and transcription inference
# all scale roughly linearly with audio length.
_WEIGHT_CLEAN_PER_MIN = 1.5
_WEIGHT_DIARIZE_PER_MIN = 3.0
_WEIGHT_TRANSCRIBE_PER_MIN = 5.0


def _compute_weights(
    duration_min: float,
    is_video: bool,
    clean_audio: bool,
) -> dict[str, float]:
    """Build per-stage weights scaled by audio duration."""
    w: dict[str, float] = {}
    if is_video:
        w["extract_video"] = _WEIGHT_EXTRACT_VIDEO
    w["load_audio"] = _WEIGHT_LOAD_AUDIO
    if clean_audio:
        w["clean"] = 2.0 + _WEIGHT_CLEAN_PER_MIN * duration_min
    w["diarize"] = _WEIGHT_LOAD_DIARIZE_MODEL + _WEIGHT_DIARIZE_PER_MIN * duration_min
    w["transcribe"] = _WEIGHT_LOAD_TRANSCRIBE_MODEL + _WEIGHT_TRANSCRIBE_PER_MIN * duration_min
    w["align"] = _WEIGHT_ALIGN
    w["write"] = _WEIGHT_WRITE
    return w


class _StageTracker:
    """Maps per-stage local progress (0-1) to overall pipeline progress (0-1).

    Thread-safe: multiple parallel reporters can call concurrently.
    """

    def __init__(self, callback: ProgressCallback, total_weight: float) -> None:
        self._callback = callback
        self._total_weight = total_weight
        self._completed_weight = 0.0
        self._current_weight = 0.0
        self._lock = threading.Lock()

    def stage(self, weight: float) -> Callable[[str, float], None]:
        """Begin a new sequential stage with the given weight.

        Returns a reporter accepting (description, local_fraction 0.0-1.0).
        """
        with self._lock:
            self._completed_weight += self._current_weight
            self._current_weight = weight
            base = self._completed_weight
            w = weight
            total = self._total_weight

        def report(description: str, local_fraction: float) -> None:
            overall = (base + w * local_fraction) / total
            self._callback(description, min(overall, 1.0))

        return report

    def parallel_stages(self, weights: dict[str, float]) -> dict[str, Callable[[str, float], None]]:
        """Begin multiple stages that will run concurrently.

        The combined weight of all parallel stages occupies a single range in
        the overall progress bar. Overall progress within the range is the
        weighted average of each branch's local fraction, so the bar advances
        smoothly as either branch makes progress.

        Returns a dict mapping each key to its reporter.
        """
        combined_weight = sum(weights.values())
        with self._lock:
            self._completed_weight += self._current_weight
            self._current_weight = combined_weight
            base = self._completed_weight
            total = self._total_weight

        # Track each parallel branch's local fraction
        local_fracs: dict[str, float] = {k: 0.0 for k in weights}
        frac_lock = threading.Lock()
        last_desc = ""

        def _make_reporter(key: str) -> Callable[[str, float], None]:
            def report(description: str, local_fraction: float) -> None:
                nonlocal last_desc
                with frac_lock:
                    local_fracs[key] = local_fraction
                    last_desc = description
                    # Weighted average of all parallel branches
                    weighted_sum = sum(local_fracs[k] * weights[k] for k in weights)
                    overall = (base + weighted_sum) / total
                self._callback(last_desc, min(overall, 1.0))

            return report

        return {k: _make_reporter(k) for k in weights}


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
    on_progress: ProgressCallback | None = None,
    parallel: bool = True,
) -> PipelineResult:
    """
    Run the full diarized transcription pipeline.

    Args:
        input_path: Path to input audio or video file
        hf_token: Hugging Face token for pyannote models
        config: Pipeline configuration (uses defaults if None)
        save_cleaned: Save cleaned audio as .cleaned.wav
        on_progress: Callback receiving (stage_description, overall_fraction)
            where overall_fraction smoothly advances from 0.0 to 1.0.
            Stage weights are scaled by audio duration so that the bar
            reflects estimated wall-clock cost.
        parallel: Run diarization and transcription concurrently (default True).
            Disable if GPU memory is limited.
    """
    if config is None:
        config = PipelineConfig()

    out_base = input_path.with_suffix("")
    is_vid = audio.is_video(input_path)
    extracted_tmp: Path | None = None

    # --- Extract audio from video (if needed) ---
    if is_vid:
        extracted_tmp = audio.extract_audio(input_path)
        load_path = extracted_tmp
    else:
        load_path = input_path

    # --- Load audio ---
    audio_data, sr = audio.load_mono_16k(load_path)

    # Now we know the duration — build duration-aware weights
    duration_min = (len(audio_data) / sr) / 60.0
    weights = _compute_weights(duration_min, is_vid, config.clean_audio)
    total_weight = sum(weights.values())

    # Set up tracker, marking extract + load as already completed
    tracker: _StageTracker | None = None
    if on_progress:
        tracker = _StageTracker(on_progress, total_weight)

        if is_vid:
            report = tracker.stage(weights["extract_video"])
            report("Extracting audio from video", 1.0)
        report = tracker.stage(weights["load_audio"])
        report("Loading audio", 1.0)

    # --- Clean audio ---
    if config.clean_audio:
        stage_report = tracker.stage(weights["clean"]) if tracker else None
        audio_data = audio.clean(
            audio_data,
            sr,
            highpass_freq=config.highpass_freq,
            noise_reduction_strength=config.noise_reduction_strength,
            on_progress=stage_report,
        )
        if save_cleaned:
            cleaned_path = out_base.with_suffix(".cleaned.wav")
            sf.write(str(cleaned_path), audio_data, sr)

    # Save temp WAV for diarization
    tmp_wav = out_base.with_suffix(".tmp_16k_mono.wav")
    sf.write(str(tmp_wav), audio_data, sr)

    try:
        if parallel:
            turns, words = _run_diarize_transcribe_parallel(
                tmp_wav,
                audio_data,
                sr,
                hf_token,
                config,
                tracker,
                weights,
            )
        else:
            turns, words = _run_diarize_transcribe_sequential(
                tmp_wav,
                audio_data,
                sr,
                hf_token,
                config,
                tracker,
                weights,
            )

        # --- Align ---
        if tracker:
            report = tracker.stage(weights["align"])
            report("Aligning words to speakers", 0.0)
        utterances = align_words_to_speakers(
            turns,
            words,
            max_gap_s=config.max_gap_s,
            gap_tolerance=config.gap_tolerance,
        )
        if tracker:
            report("Aligning words to speakers", 1.0)

        # --- Write outputs ---
        if tracker:
            report = tracker.stage(weights["write"])
            report("Writing outputs", 0.0)
        output_paths = write_all(out_base, utterances)
        if tracker:
            report("Writing outputs", 1.0)

    finally:
        tmp_wav.unlink(missing_ok=True)
        if extracted_tmp is not None:
            extracted_tmp.unlink(missing_ok=True)

    return PipelineResult(
        utterances=utterances,
        turns=turns,
        words=words,
        output_paths=output_paths,
    )


def _run_diarize_transcribe_parallel(
    tmp_wav: Path,
    audio_data: NDArray[np.floating],
    sr: int,
    hf_token: str,
    config: PipelineConfig,
    tracker: _StageTracker | None,
    weights: dict[str, float],
) -> tuple[list[SpeakerTurn], list[WordItem]]:
    """Run diarization and transcription concurrently."""
    if tracker:
        reporters = tracker.parallel_stages(
            {
                "diarize": weights["diarize"],
                "transcribe": weights["transcribe"],
            }
        )
        diarize_report = reporters["diarize"]
        transcribe_report = reporters["transcribe"]
    else:
        diarize_report = None
        transcribe_report = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        diarize_future = pool.submit(
            diarization.diarize,
            tmp_wav,
            token=hf_token,
            model_id=config.diarization_model,
            num_speakers=config.num_speakers,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
            on_progress=diarize_report,
        )
        transcribe_future = pool.submit(
            transcription.transcribe,
            audio_data,
            sr,
            model_size=config.whisper_model,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
            on_progress=transcribe_report,
        )

        turns = diarize_future.result()
        words = transcribe_future.result()

    return turns, words


def _run_diarize_transcribe_sequential(
    tmp_wav: Path,
    audio_data: NDArray[np.floating],
    sr: int,
    hf_token: str,
    config: PipelineConfig,
    tracker: _StageTracker | None,
    weights: dict[str, float],
) -> tuple[list[SpeakerTurn], list[WordItem]]:
    """Run diarization then transcription sequentially."""
    stage_report = tracker.stage(weights["diarize"]) if tracker else None
    turns = diarization.diarize(
        tmp_wav,
        token=hf_token,
        model_id=config.diarization_model,
        num_speakers=config.num_speakers,
        min_speakers=config.min_speakers,
        max_speakers=config.max_speakers,
        on_progress=stage_report,
    )

    stage_report = tracker.stage(weights["transcribe"]) if tracker else None
    words = transcription.transcribe(
        audio_data,
        sr,
        model_size=config.whisper_model,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
        on_progress=stage_report,
    )

    return turns, words
