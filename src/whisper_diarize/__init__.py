"""
whisper-diarize: Local diarized transcription using Qwen3-ASR and Nemotron.

Example usage:
    from pathlib import Path
    from whisper_diarize import run, PipelineConfig

    config = PipelineConfig(profile="accuracy", device="cuda")
    result = run(Path("audio.mp3"), config=config)

    for u in result.utterances:
        print(f"{u.speaker}: {u.text}")
"""

__version__ = "0.1.0"

from whisper_diarize.models import SpeakerTurn, Utterance, WordItem
from whisper_diarize.pipeline import PipelineConfig, PipelineResult, run

__all__ = [
    "__version__",
    # Main API
    "run",
    "PipelineConfig",
    "PipelineResult",
    # Models
    "SpeakerTurn",
    "Utterance",
    "WordItem",
]
