"""CLI authentication follows the selected diarization backend."""

from unittest.mock import patch

import pytest

from whisper_diarize.cli import main
from whisper_diarize.pipeline import PipelineResult


def test_default_accuracy_can_run_without_a_hugging_face_token(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.wav"
    audio.touch()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["transcribe", str(audio), "--device", "cpu"])
    with (
        patch("whisper_diarize.cli.load_dotenv"),
        patch("whisper_diarize.pipeline.run", return_value=PipelineResult([], [], [])) as run,
    ):
        main()
    assert run.call_args.kwargs["config"].diarization_model == "nvidia/Nemotron-3-Diarization"
    assert run.call_args.kwargs["hf_token"] == ""


def test_pyannote_still_requires_a_hugging_face_token(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.wav"
    audio.touch()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["transcribe", str(audio), "--num-speakers", "2", "--device", "cpu"]
    )
    with patch("whisper_diarize.cli.load_dotenv"), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
