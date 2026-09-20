"""Tests for transcription model resolution."""

from __future__ import annotations

import pytest

from whisper_diarize.models import WordItem
from whisper_diarize.transcription import is_qwen_asr_model, resolve_model_id, words_from_transcript


class TestResolveModelId:
    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("large-v3", "large-v3"),
            ("openai/whisper-large-v3", "large-v3"),
            ("large-v3-turbo", "large-v3-turbo"),
            ("openai/whisper-large-v3-turbo", "large-v3-turbo"),
            ("distil-large-v3", "distil-large-v3"),
            ("distil-whisper/distil-large-v3", "distil-large-v3"),
            ("distil-large-v3.5", "distil-large-v3.5"),
            ("distil-whisper/distil-large-v3.5", "distil-large-v3.5"),
        ],
    )
    def test_maps_known_aliases(self, model_name: str, expected: str):
        assert resolve_model_id(model_name) == expected

    def test_passthrough_for_explicit_hf_id(self):
        model_id = "my-org/custom-whisper-model"
        assert resolve_model_id(model_id) == model_id

    def test_rejects_empty_model_name(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_model_id("   ")

    def test_qwen_hf_id_passthrough(self):
        assert resolve_model_id("Qwen/Qwen3-ASR-1.7B-hf") == "Qwen/Qwen3-ASR-1.7B-hf"


class TestIsQwenAsrModel:
    def test_detects_qwen3_asr(self):
        assert is_qwen_asr_model("Qwen/Qwen3-ASR-1.7B-hf") is True
        assert is_qwen_asr_model("qwen/qwen3-asr-1.7b-hf") is True

    def test_rejects_whisper_models(self):
        assert is_qwen_asr_model("large-v3") is False
        assert is_qwen_asr_model("openai/whisper-large-v3") is False


class TestWordsFromTranscript:
    def test_spreads_words_across_the_turn(self):
        words = words_from_transcript("hello there chris", start_s=10.0, end_s=13.0)
        assert [w.word for w in words] == ["hello", "there", "chris"]
        assert words[0].start_s == pytest.approx(10.0)
        assert words[-1].end_s == pytest.approx(13.0)
        assert words[0].end_s <= words[1].start_s

    def test_empty_text_returns_no_words(self):
        assert words_from_transcript("   ", start_s=0.0, end_s=1.0) == []

    def test_single_word_covers_full_span(self):
        words = words_from_transcript("yes", start_s=2.0, end_s=2.4)
        assert words == [WordItem(start_s=2.0, end_s=2.4, word="yes")]
