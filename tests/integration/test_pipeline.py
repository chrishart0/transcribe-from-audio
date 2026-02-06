"""Integration tests for transcription output validation.

Uses a sample fixture with 2 speakers to validate output structure and formats.
The fixture represents a simple conversation between SPEAKER_00 and SPEAKER_01.

To run integration tests: uv run pytest tests/integration -v
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from whisper_diarize.models import Utterance
from whisper_diarize.output import to_json, to_srt, to_text, write_all

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_JSON = FIXTURES_DIR / "sample_transcript.json"


@pytest.mark.integration
class TestTranscriptStructure:
    """Validate transcript output structure."""

    @pytest.fixture
    def json_output(self) -> list[dict]:
        if not SAMPLE_JSON.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_JSON}")
        return json.loads(SAMPLE_JSON.read_text())

    def test_json_has_required_fields(self, json_output: list[dict]):
        """Each utterance must have start, end, speaker, and text."""
        required_fields = {"start", "end", "speaker", "text"}

        for i, utterance in enumerate(json_output):
            missing = required_fields - set(utterance.keys())
            assert not missing, f"Utterance {i} missing fields: {missing}"

    def test_json_has_multiple_speakers(self, json_output: list[dict]):
        """Output should contain at least 2 distinct speakers."""
        speakers = {u["speaker"] for u in json_output}
        real_speakers = speakers - {"UNKNOWN"}
        assert len(real_speakers) >= 2, f"Expected at least 2 speakers, got: {real_speakers}"

    def test_json_timestamps_are_ordered(self, json_output: list[dict]):
        """Utterances should be in chronological order."""
        for i in range(1, len(json_output)):
            prev_start = json_output[i - 1]["start"]
            curr_start = json_output[i]["start"]
            assert curr_start >= prev_start, (
                f"Utterance {i} starts at {curr_start} before utterance {i - 1} at {prev_start}"
            )

    def test_json_timestamps_are_valid(self, json_output: list[dict]):
        """Each utterance end time must be >= start time."""
        for i, utterance in enumerate(json_output):
            assert utterance["end"] >= utterance["start"], (
                f"Utterance {i} has end ({utterance['end']}) before start ({utterance['start']})"
            )

    def test_json_speakers_are_strings(self, json_output: list[dict]):
        """Speaker labels must be non-empty strings."""
        for i, utterance in enumerate(json_output):
            assert isinstance(utterance["speaker"], str), f"Utterance {i} speaker not a string"
            assert utterance["speaker"], f"Utterance {i} has empty speaker"

    def test_json_text_is_non_empty(self, json_output: list[dict]):
        """Text content must be non-empty."""
        for i, utterance in enumerate(json_output):
            assert isinstance(utterance["text"], str), f"Utterance {i} text not a string"
            assert utterance["text"].strip(), f"Utterance {i} has empty text"

    def test_speaker_turns_alternate(self, json_output: list[dict]):
        """Verify that there are speaker transitions (not all same speaker)."""
        if len(json_output) < 2:
            pytest.skip("Need at least 2 utterances to check alternation")

        transitions = sum(
            1
            for i in range(1, len(json_output))
            if json_output[i]["speaker"] != json_output[i - 1]["speaker"]
        )
        assert transitions > 0, "Expected at least one speaker transition"


@pytest.mark.integration
class TestOutputFormatters:
    """Test output formatters produce valid output."""

    @pytest.fixture
    def utterances(self) -> list[Utterance]:
        """Create test utterances from fixture."""
        if not SAMPLE_JSON.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_JSON}")

        data = json.loads(SAMPLE_JSON.read_text())
        return [
            Utterance(
                start_s=u["start"],
                end_s=u["end"],
                speaker=u["speaker"],
                text=u["text"],
            )
            for u in data
        ]

    def test_to_json_roundtrip(self, utterances: list[Utterance]):
        """JSON output should be parseable and contain all utterances."""
        json_str = to_json(utterances)
        parsed = json.loads(json_str)

        assert len(parsed) == len(utterances)
        for orig, parsed_u in zip(utterances, parsed, strict=True):
            assert parsed_u["speaker"] == orig.speaker
            assert parsed_u["text"] == orig.text

    def test_to_text_contains_speakers(self, utterances: list[Utterance]):
        """Text output should contain all speaker labels."""
        text = to_text(utterances)
        speakers = {u.speaker for u in utterances}

        for speaker in speakers:
            assert speaker in text, f"Speaker {speaker} not in text output"

    def test_to_srt_valid_format(self, utterances: list[Utterance]):
        """SRT output should have valid subtitle format."""
        srt = to_srt(utterances)
        blocks = srt.strip().split("\n\n")

        assert len(blocks) == len(utterances)

        for i, block in enumerate(blocks):
            lines = block.strip().split("\n")
            assert len(lines) >= 3, f"SRT block {i} has fewer than 3 lines"
            assert lines[0] == str(i + 1), f"SRT block {i} has wrong sequence number"
            assert "-->" in lines[1], f"SRT block {i} missing timestamp arrow"

    def test_to_srt_timestamps_formatted(self, utterances: list[Utterance]):
        """SRT timestamps should be properly formatted HH:MM:SS,mmm."""
        import re

        srt = to_srt(utterances)
        timestamp_pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"

        matches = re.findall(timestamp_pattern, srt)
        assert len(matches) == len(utterances), "Each utterance should have a timestamp line"


@pytest.mark.integration
class TestWriteAllFormats:
    """Test write_all produces all output files."""

    @pytest.fixture
    def utterances(self) -> list[Utterance]:
        """Create test utterances from fixture."""
        if not SAMPLE_JSON.exists():
            pytest.skip(f"Sample file not found: {SAMPLE_JSON}")

        data = json.loads(SAMPLE_JSON.read_text())
        return [
            Utterance(
                start_s=u["start"],
                end_s=u["end"],
                speaker=u["speaker"],
                text=u["text"],
            )
            for u in data
        ]

    def test_write_all_creates_files(self, utterances: list[Utterance]):
        """write_all should create json, txt, and srt files."""
        with TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / "test_output"
            outputs = write_all(out_base, utterances)

            assert outputs["json"].exists()
            assert outputs["txt"].exists()
            assert outputs["srt"].exists()

    def test_write_all_files_have_content(self, utterances: list[Utterance]):
        """All output files should have non-empty content."""
        with TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / "test_output"
            outputs = write_all(out_base, utterances)

            for fmt, path in outputs.items():
                content = path.read_text()
                assert content.strip(), f"{fmt} file is empty"

    def test_write_all_json_matches_utterances(self, utterances: list[Utterance]):
        """JSON file should contain same number of utterances."""
        with TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / "test_output"
            outputs = write_all(out_base, utterances)

            data = json.loads(outputs["json"].read_text())
            assert len(data) == len(utterances)
