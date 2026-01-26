"""Speaker diarization using pyannote.audio."""

from __future__ import annotations

from pathlib import Path

from whisper_diarize.models import SpeakerTurn


def diarize(
    wav_path: Path,
    token: str,
    model_id: str = "pyannote/speaker-diarization-3.1",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """
    Run speaker diarization on audio file.

    Args:
        wav_path: Path to WAV file (16kHz mono recommended)
        token: Hugging Face token for pyannote models
        model_id: Diarization model to use
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum expected speakers
        max_speakers: Maximum expected speakers
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_id, token=token)

    diarization_args = {}
    if num_speakers is not None:
        diarization_args["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarization_args["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarization_args["max_speakers"] = max_speakers

    result = pipeline(str(wav_path), **diarization_args)

    turns: list[SpeakerTurn] = []
    for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start_s=float(turn.start),
                end_s=float(turn.end),
                speaker=str(speaker),
            )
        )

    turns.sort(key=lambda t: t.start_s)
    return turns

