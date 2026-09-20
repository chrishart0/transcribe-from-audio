"""Tests for transcription model resolution."""

from __future__ import annotations

import pytest

from whisper_diarize.transcription import resolve_model_id


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
