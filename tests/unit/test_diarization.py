"""Exercise the Nemotron backend through the public diarization entry point."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from whisper_diarize.diarization import diarize


def test_nemotron_preserves_short_and_overlapping_speaker_turns(tmp_path):
    path = tmp_path / "conversation.wav"
    sf.write(path, np.zeros(16000), 16000)
    inputs = MagicMock()
    inputs.to.return_value = inputs
    processor = MagicMock()
    processor.return_value = inputs
    processor.extract_speaker_dict.return_value = [
        [
            {"Start": 0.0, "End": 0.9, "Speaker": 0},
            {"Start": 0.5, "End": 0.7, "Speaker": 1},
        ]
    ]
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.device, model.dtype = "cpu", torch.float32
    model.return_value = SimpleNamespace(logits=torch.zeros(1))
    with (
        patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
        patch(
            "transformers.AutoModelForAudioFrameClassification.from_pretrained", return_value=model
        ),
        patch(
            "pyannote.audio.Pipeline.from_pretrained", side_effect=AssertionError("wrong backend")
        ),
    ):
        turns = diarize(path, token="", model_id="nvidia/Nemotron-3-Diarization", device="cpu")
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[1].end_s - turns[1].start_s == pytest.approx(0.2)
