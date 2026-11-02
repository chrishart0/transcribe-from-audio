"""Exercise transcription and measured timestamps without downloading models."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from whisper_diarize.transcription import transcribe


@pytest.fixture
def qwen_models():
    inputs = MagicMock()
    inputs.to.return_value = inputs
    inputs.__getitem__.return_value = torch.zeros((1, 1), dtype=torch.long)
    processor = MagicMock()
    processor.apply_transcription_request.return_value = inputs
    processor.decode.side_effect = lambda ids, **kwargs: ["Hello there"] * ids.shape[0]
    processor.prepare_forced_aligner_inputs.return_value = (inputs, [["Hello", "there"]])
    processor.decode_forced_alignment.return_value = [
        [
            {"text": "Hello", "start_time": 0.2, "end_time": 0.4},
            {"text": "there", "start_time": 0.6, "end_time": 0.9},
        ]
    ]
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.device = "cpu"
    model.dtype = torch.float32
    model.return_value = SimpleNamespace(logits=torch.zeros(1))

    def generate(**kwargs):
        request = processor.apply_transcription_request.call_args.kwargs["audio"]
        count = len(request) if isinstance(request, list) else 1
        return torch.zeros((count, 2), dtype=torch.long)

    model.generate.side_effect = generate
    with (
        patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
        patch("transformers.AutoModelForMultimodalLM.from_pretrained", return_value=model),
        patch("transformers.AutoModelForTokenClassification.from_pretrained", return_value=model),
    ):
        yield processor
    from whisper_diarize import qwen_asr

    qwen_asr._load.cache_clear()


def test_qwen_uses_measured_word_times_across_chunks(qwen_models):
    words = transcribe(
        np.zeros(45 * 16000, dtype=np.float32),
        16000,
        model_size="Qwen/Qwen3-ASR-1.7B-hf",
        device="cpu",
    )
    assert [w.word for w in words] == ["Hello", "there"] * 3
    assert [w.start_s for w in words] == pytest.approx([0.2, 0.6, 20.2, 20.6, 40.2, 40.6])
    assert [w.end_s for w in words] == pytest.approx([0.4, 0.9, 20.4, 20.9, 40.4, 40.9])


def test_qwen_empty_audio_does_not_load_models():
    with patch("transformers.AutoProcessor.from_pretrained") as load:
        assert (
            transcribe(
                np.zeros(0, dtype=np.float32),
                16000,
                model_size="Qwen/Qwen3-ASR-1.7B-hf",
                device="cpu",
            )
            == []
        )
    load.assert_not_called()


def test_qwen_silence_has_no_invented_timestamps(qwen_models):
    qwen_models.decode.side_effect = None
    qwen_models.decode.return_value = [""]
    assert (
        transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            model_size="Qwen/Qwen3-ASR-1.7B-hf",
            device="cpu",
        )
        == []
    )


def test_qwen_cuda_batches_audio(qwen_models):
    words = transcribe(
        np.zeros(85 * 16000, dtype=np.float32),
        16000,
        model_size="Qwen/Qwen3-ASR-1.7B-hf",
        device="cuda",
    )
    assert len(words) == 10
    assert words[-1].start_s == pytest.approx(80.6)
    requests = qwen_models.apply_transcription_request.call_args_list
    assert [len(call.kwargs["audio"]) for call in requests] == [4, 1]
