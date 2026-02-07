"""Tests for output module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisper_diarize.models import Utterance
from whisper_diarize.output import format_srt_timestamp, to_json, to_srt, to_text, write_all


class TestFormatSrtTimestamp:
    def test_zero(self):
        assert format_srt_timestamp(0.0) == "00:00:00,000"

    def test_fractional_seconds(self):
        assert format_srt_timestamp(1.234) == "00:00:01,234"

    def test_minutes(self):
        assert format_srt_timestamp(65.5) == "00:01:05,500"

    def test_hours(self):
        assert format_srt_timestamp(3661.123) == "01:01:01,123"

    def test_rounding(self):
        assert format_srt_timestamp(1.2346) == "00:00:01,235"


class TestFormatters:
    @pytest.fixture
    def sample_utterances(self) -> list[Utterance]:
        return [
            Utterance(start_s=0.0, end_s=1.0, speaker="A", text="Hello"),
            Utterance(start_s=1.5, end_s=2.5, speaker="B", text="World"),
        ]

    def test_to_json(self, sample_utterances: list[Utterance]):
        result = json.loads(to_json(sample_utterances))
        assert len(result) == 2
        assert result[0]["speaker"] == "A"
        assert result[0]["text"] == "Hello"

    def test_to_text(self, sample_utterances: list[Utterance]):
        result = to_text(sample_utterances)
        assert "A: Hello" in result
        assert "B: World" in result

    def test_to_srt(self, sample_utterances: list[Utterance]):
        result = to_srt(sample_utterances)
        assert "00:00:00,000 --> 00:00:01,000" in result
        assert "A: Hello" in result


class TestWriteAll:
    def test_writes_all_formats(self, tmp_path: Path):
        utterances = [
            Utterance(start_s=0.0, end_s=1.0, speaker="A", text="Hello"),
            Utterance(start_s=1.5, end_s=2.5, speaker="B", text="World"),
        ]
        out_base = tmp_path / "test"
        outputs = write_all(out_base, utterances)

        assert outputs["json"].exists()
        assert outputs["txt"].exists()
        assert outputs["srt"].exists()

        data = json.loads(outputs["json"].read_text())
        assert len(data) == 2

    def test_multi_dot_filename(self, tmp_path: Path):
        out_base = tmp_path / "my.recording"
        utterances = [
            Utterance(start_s=0.0, end_s=1.0, speaker="A", text="Hello"),
        ]
        outputs = write_all(out_base, utterances)

        assert outputs["json"].name == "my.recording.diarized.json"
        assert outputs["txt"].name == "my.recording.diarized.txt"
        assert outputs["srt"].name == "my.recording.diarized.srt"
        for p in outputs.values():
            assert p.exists()

    def test_empty_utterances(self, tmp_path: Path):
        out_base = tmp_path / "empty"
        outputs = write_all(out_base, [])

        assert outputs["json"].read_text() == "[]"
        assert outputs["txt"].read_text() == "\n"
