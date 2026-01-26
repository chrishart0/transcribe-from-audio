"""
sales-agent: Diarized transcription using Whisper + pyannote.audio.

Example usage:
    from sales_agent import pipeline, PipelineConfig

    config = PipelineConfig(whisper_model="large-v3", device="cuda")
    result = pipeline.run(Path("audio.mp3"), hf_token="hf_...", config=config)

    for u in result.utterances:
        print(f"{u.speaker}: {u.text}")
"""

from sales_agent.models import SpeakerTurn, Utterance, WordItem
from sales_agent.pipeline import PipelineConfig, PipelineResult

__all__ = [
    # Models
    "SpeakerTurn",
    "Utterance",
    "WordItem",
    # Pipeline
    "PipelineConfig",
    "PipelineResult",
]
