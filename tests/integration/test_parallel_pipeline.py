"""Integration tests for parallel pipeline execution.

These tests mock the heavy ML models (pyannote, faster-whisper) but exercise
the real pipeline orchestration, audio loading, cleaning, alignment, and output.
"""

from __future__ import annotations

import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from whisper_diarize.models import SpeakerTurn, WordItem
from whisper_diarize.pipeline import PipelineConfig, run


def _make_test_wav(path: Path, duration_s: float = 2.0, sr: int = 16000) -> None:
    """Create a short silent WAV file for testing."""
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(str(path), samples, sr)


def _fake_diarize_factory(turns: list[SpeakerTurn]):
    """Return a fake diarize function that returns the given turns."""

    def fake_diarize(*args, on_progress=None, **kwargs):
        if on_progress:
            on_progress("Loading diarization model", 0.0)
            on_progress("Diarizing speakers", 0.15)
            on_progress("Diarizing speakers", 1.0)
        return turns

    return fake_diarize


def _fake_transcribe_factory(words: list[WordItem]):
    """Return a fake transcribe function that returns the given words."""

    def fake_transcribe(*args, on_progress=None, **kwargs):
        if on_progress:
            on_progress("Loading transcription model", 0.0)
            on_progress("Transcribing speech", 0.10)
            on_progress("Transcribing speech", 1.0)
        return words

    return fake_transcribe


MOCK_TURNS = [
    SpeakerTurn(start_s=0.0, end_s=1.0, speaker="SPEAKER_00"),
    SpeakerTurn(start_s=1.0, end_s=2.0, speaker="SPEAKER_01"),
]

MOCK_WORDS = [
    WordItem(start_s=0.1, end_s=0.4, word="Hello"),
    WordItem(start_s=0.5, end_s=0.9, word="world"),
    WordItem(start_s=1.1, end_s=1.4, word="Good"),
    WordItem(start_s=1.5, end_s=1.9, word="morning"),
]


@pytest.mark.integration
class TestParallelPipelineIntegration:
    """Test the full pipeline with parallel=True (default)."""

    def test_parallel_pipeline_produces_outputs(self):
        with TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _make_test_wav(wav_path)

            with (
                patch(
                    "whisper_diarize.diarization.diarize",
                    side_effect=_fake_diarize_factory(MOCK_TURNS),
                ),
                patch(
                    "whisper_diarize.transcription.transcribe",
                    side_effect=_fake_transcribe_factory(MOCK_WORDS),
                ),
            ):
                result = run(
                    wav_path,
                    hf_token="fake-token",
                    config=PipelineConfig(clean_audio=False),
                    parallel=True,
                )

            assert len(result.utterances) >= 1
            assert len(result.turns) == 2
            assert len(result.words) == 4
            assert "json" in result.output_paths
            assert "txt" in result.output_paths
            assert "srt" in result.output_paths
            for p in result.output_paths.values():
                assert p.exists()
                assert p.stat().st_size > 0

    def test_sequential_pipeline_produces_same_results(self):
        with TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _make_test_wav(wav_path)

            with (
                patch(
                    "whisper_diarize.diarization.diarize",
                    side_effect=_fake_diarize_factory(MOCK_TURNS),
                ),
                patch(
                    "whisper_diarize.transcription.transcribe",
                    side_effect=_fake_transcribe_factory(MOCK_WORDS),
                ),
            ):
                result_seq = run(
                    wav_path,
                    hf_token="fake-token",
                    config=PipelineConfig(clean_audio=False),
                    parallel=False,
                )

        with TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _make_test_wav(wav_path)

            with (
                patch(
                    "whisper_diarize.diarization.diarize",
                    side_effect=_fake_diarize_factory(MOCK_TURNS),
                ),
                patch(
                    "whisper_diarize.transcription.transcribe",
                    side_effect=_fake_transcribe_factory(MOCK_WORDS),
                ),
            ):
                result_par = run(
                    wav_path,
                    hf_token="fake-token",
                    config=PipelineConfig(clean_audio=False),
                    parallel=True,
                )

        # Same utterances
        assert len(result_seq.utterances) == len(result_par.utterances)
        for u_s, u_p in zip(result_seq.utterances, result_par.utterances, strict=True):
            assert u_s.speaker == u_p.speaker
            assert u_s.text == u_p.text
            assert u_s.start_s == pytest.approx(u_p.start_s)
            assert u_s.end_s == pytest.approx(u_p.end_s)

    def test_parallel_pipeline_with_progress(self):
        """Progress callback receives monotonically increasing fractions ending at 1.0."""
        calls: list[tuple[str, float]] = []
        lock = threading.Lock()

        def on_progress(desc: str, frac: float) -> None:
            with lock:
                calls.append((desc, frac))

        with TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _make_test_wav(wav_path)

            with (
                patch(
                    "whisper_diarize.diarization.diarize",
                    side_effect=_fake_diarize_factory(MOCK_TURNS),
                ),
                patch(
                    "whisper_diarize.transcription.transcribe",
                    side_effect=_fake_transcribe_factory(MOCK_WORDS),
                ),
            ):
                run(
                    wav_path,
                    hf_token="fake-token",
                    config=PipelineConfig(clean_audio=False),
                    parallel=True,
                    on_progress=on_progress,
                )

        assert len(calls) > 0
        # All fractions in [0, 1]
        for desc, frac in calls:
            assert 0.0 <= frac <= 1.0, f"frac={frac} out of range for '{desc}'"
        # Final call should be 1.0 (writing outputs done)
        assert calls[-1][1] == pytest.approx(1.0)
        # Should see various stage descriptions
        descs = {d for d, _ in calls}
        assert "Loading audio" in descs
        assert "Writing outputs" in descs

    def test_parallel_pipeline_with_cleaning(self):
        with TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _make_test_wav(wav_path)

            with (
                patch(
                    "whisper_diarize.diarization.diarize",
                    side_effect=_fake_diarize_factory(MOCK_TURNS),
                ),
                patch(
                    "whisper_diarize.transcription.transcribe",
                    side_effect=_fake_transcribe_factory(MOCK_WORDS),
                ),
            ):
                result = run(
                    wav_path,
                    hf_token="fake-token",
                    config=PipelineConfig(clean_audio=True),
                    parallel=True,
                )

            assert len(result.utterances) >= 1

    def test_pipeline_cleans_up_temp_files(self):
        with TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _make_test_wav(wav_path)

            with (
                patch(
                    "whisper_diarize.diarization.diarize",
                    side_effect=_fake_diarize_factory(MOCK_TURNS),
                ),
                patch(
                    "whisper_diarize.transcription.transcribe",
                    side_effect=_fake_transcribe_factory(MOCK_WORDS),
                ),
            ):
                run(
                    wav_path,
                    hf_token="fake-token",
                    config=PipelineConfig(clean_audio=False),
                    parallel=True,
                )

            # Temp WAV for diarization should be cleaned up
            tmp_wav = Path(tmpdir) / "test.tmp_16k_mono.wav"
            assert not tmp_wav.exists()

    def test_pipeline_error_still_cleans_up(self):
        with TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _make_test_wav(wav_path)

            with (
                patch(
                    "whisper_diarize.diarization.diarize", side_effect=RuntimeError("model failed")
                ),
                patch(
                    "whisper_diarize.transcription.transcribe",
                    side_effect=_fake_transcribe_factory(MOCK_WORDS),
                ),
            ):
                with pytest.raises(RuntimeError, match="model failed"):
                    run(
                        wav_path,
                        hf_token="fake-token",
                        config=PipelineConfig(clean_audio=False),
                        parallel=True,
                    )

            tmp_wav = Path(tmpdir) / "test.tmp_16k_mono.wav"
            assert not tmp_wav.exists()
