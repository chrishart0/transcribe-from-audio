"""Tests for transcription model resolution."""

from __future__ import annotations

import pytest

from whisper_diarize.transcription import resolve_model_id


class TestResolveModelId:
    def test_maps_known_aliases(self):
        assert resolve_model_id("large-v3") == "large-v3"
        assert resolve_model_id("openai/whisper-large-v3") == "large-v3"
        assert resolve_model_id("large-v3-turbo") == "large-v3-turbo"
        assert resolve_model_id("distil-whisper/distil-large-v3.5") == "distil-large-v3.5"

    def test_passthrough_for_explicit_hf_id(self):
        model_id = "my-org/custom-whisper-model"
        assert resolve_model_id(model_id) == model_id

    def test_rejects_empty_model_name(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_model_id("   ")
