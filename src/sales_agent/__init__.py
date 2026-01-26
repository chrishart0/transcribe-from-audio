"""
sales-agent: Diarized transcription using Whisper + pyannote.audio.

Example usage:
    from sales_agent import run, PipelineConfig

    config = PipelineConfig(whisper_model="large-v3", device="cuda")
    result = run(Path("audio.mp3"), hf_token="hf_...", config=config)

    for u in result.utterances:
        print(f"{u.speaker}: {u.text}")
"""

__version__ = "0.1.0"

from sales_agent.models import SpeakerTurn, Utterance, WordItem
from sales_agent.pipeline import PipelineConfig, PipelineResult, run

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
