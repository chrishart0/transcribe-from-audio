"""Speaker diarization using pyannote.audio."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from whisper_diarize.models import SpeakerTurn

if TYPE_CHECKING:
    from whisper_diarize.audio import StageProgressFn


def diarize(
    wav_path: Path,
    token: str,
    model_id: str = "pyannote/speaker-diarization-community-1",
    device: str = "cuda",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    clustering_threshold: float | None = None,
    min_duration_off: float | None = None,
    on_progress: StageProgressFn | None = None,
) -> list[SpeakerTurn]:
    """
    Run speaker diarization on audio file.

    Args:
        wav_path: Path to WAV file (16kHz mono recommended)
        token: Hugging Face token for pyannote models
        model_id: Diarization model to use
        device: Device for inference ("cuda" or "cpu")
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum expected speakers
        max_speakers: Maximum expected speakers
        clustering_threshold: Override agglomerative clustering threshold
        min_duration_off: Override minimum inter-turn silence duration
        on_progress: Reports (description, local_fraction) within this stage.
            Model loading is ~15% of the cost, inference is ~85%.
    """
    from pyannote.audio import Pipeline

    if on_progress:
        on_progress("Loading diarization model", 0.0)
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)  # type: ignore[call-arg]
    except Exception as exc:
        raise RuntimeError(
            "Failed to load diarization model. Verify --hf-token has access to "
            f"'{model_id}' and that model terms are accepted on Hugging Face."
        ) from exc
    if pipeline is None:
        raise RuntimeError(f"Failed to load diarization model: {model_id}")

    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception:
            # Fall back silently to CPU when CUDA setup is incomplete.
            pass

    if clustering_threshold is not None or min_duration_off is not None:
        params = pipeline.parameters(instantiated=True)
        if clustering_threshold is not None:
            params["clustering"]["threshold"] = clustering_threshold
        if min_duration_off is not None:
            params["segmentation"]["min_duration_off"] = min_duration_off
        pipeline.instantiate(params)

    diarization_args: dict[str, int] = {}
    if num_speakers is not None:
        diarization_args["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarization_args["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarization_args["max_speakers"] = max_speakers

    if on_progress:
        on_progress("Diarizing speakers", 0.15)
    try:
        result = pipeline(str(wav_path), **diarization_args)  # type: ignore[arg-type]
    except Exception as exc:
        raise RuntimeError("Diarization inference failed.") from exc

    # Prefer exclusive (non-overlapping) segments for cleaner alignment with
    # Whisper's single text stream. Falls back to overlapping then raw result.
    annotation = cast(
        Any,
        getattr(
            result,
            "exclusive_speaker_diarization",
            getattr(result, "speaker_diarization", result),
        ),
    )

    turns: list[SpeakerTurn] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start_s=float(turn.start),
                end_s=float(turn.end),
                speaker=str(speaker),
            )
        )

    turns.sort(key=lambda t: t.start_s)

    if on_progress:
        on_progress("Diarizing speakers", 1.0)

    return turns
