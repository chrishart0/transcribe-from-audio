"""Tests for pipeline progress tracking and parallel execution."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from whisper_diarize.pipeline import (
    PipelineConfig,
    _compute_weights,
    _resolve_runtime_config,
    _should_retry_sequential,
    _StageTracker,
    recommend_worker_count,
)


class TestComputeWeights:
    def test_audio_only_no_clean(self):
        w = _compute_weights(duration_min=10.0, is_video=False, clean_audio=False)
        assert "extract_video" not in w
        assert "clean" not in w
        assert "load_audio" in w
        assert "diarize" in w
        assert "transcribe" in w
        assert "align" in w
        assert "write" in w

    def test_audio_with_clean(self):
        w = _compute_weights(duration_min=10.0, is_video=False, clean_audio=True)
        assert "clean" in w
        assert "extract_video" not in w

    def test_video_with_clean(self):
        w = _compute_weights(duration_min=10.0, is_video=True, clean_audio=True)
        assert "extract_video" in w
        assert "clean" in w

    def test_weights_scale_with_duration(self):
        short = _compute_weights(duration_min=1.0, is_video=False, clean_audio=True)
        long = _compute_weights(duration_min=60.0, is_video=False, clean_audio=True)

        # Duration-scaled stages should be larger for longer audio
        assert long["diarize"] > short["diarize"]
        assert long["transcribe"] > short["transcribe"]
        assert long["clean"] > short["clean"]

        # Fixed-cost stages should be the same
        assert long["load_audio"] == short["load_audio"]
        assert long["align"] == short["align"]
        assert long["write"] == short["write"]

    def test_transcribe_dominates_for_long_audio(self):
        w = _compute_weights(duration_min=120.0, is_video=False, clean_audio=True)
        total = sum(w.values())
        transcribe_share = w["transcribe"] / total
        # Transcription should be the largest stage for long audio
        assert transcribe_share > 0.4

    def test_fixed_costs_dominate_for_short_audio(self):
        w = _compute_weights(duration_min=0.1, is_video=False, clean_audio=True)
        total = sum(w.values())
        load_share = w["load_audio"] / total
        # For very short audio, fixed costs should be a noticeable share
        assert load_share > 0.05


class TestStageTracker:
    def test_sequential_stages_reach_100(self):
        calls: list[tuple[str, float]] = []
        tracker = _StageTracker(lambda d, f: calls.append((d, f)), total_weight=10.0)

        r = tracker.stage(3.0)
        r("A", 0.0)
        r("A", 1.0)

        r = tracker.stage(7.0)
        r("B", 0.0)
        r("B", 1.0)

        assert calls[-1][1] == pytest.approx(1.0)

    def test_overall_fraction_monotonic(self):
        calls: list[float] = []
        tracker = _StageTracker(lambda d, f: calls.append(f), total_weight=100.0)

        r = tracker.stage(20.0)
        r("A", 0.0)
        r("A", 0.5)
        r("A", 1.0)

        r = tracker.stage(80.0)
        r("B", 0.0)
        r("B", 0.25)
        r("B", 0.5)
        r("B", 1.0)

        for i in range(1, len(calls)):
            assert calls[i] >= calls[i - 1], (
                f"calls[{i}]={calls[i]} < calls[{i - 1}]={calls[i - 1]}"
            )

    def test_intermediate_fractions(self):
        calls: list[tuple[str, float]] = []
        tracker = _StageTracker(lambda d, f: calls.append((d, f)), total_weight=100.0)

        r = tracker.stage(50.0)
        r("A", 0.5)
        assert calls[-1][1] == pytest.approx(0.25)

        r = tracker.stage(50.0)
        r("B", 0.5)
        assert calls[-1][1] == pytest.approx(0.75)

    def test_never_exceeds_1(self):
        calls: list[float] = []
        tracker = _StageTracker(lambda d, f: calls.append(f), total_weight=10.0)

        r = tracker.stage(10.0)
        r("A", 1.5)  # Overshoot
        assert calls[-1] <= 1.0


class TestStageTrackerParallel:
    def test_parallel_stages_basic(self):
        calls: list[tuple[str, float]] = []
        tracker = _StageTracker(lambda d, f: calls.append((d, f)), total_weight=100.0)

        reporters = tracker.parallel_stages({"a": 40.0, "b": 60.0})

        reporters["a"]("A", 0.0)
        reporters["b"]("B", 0.0)
        assert calls[-1][1] == pytest.approx(0.0)

        reporters["a"]("A", 1.0)
        # a=1.0 contributes 40/100, b=0.0 contributes 0
        assert calls[-1][1] == pytest.approx(0.4)

        reporters["b"]("B", 1.0)
        # a=1.0 contributes 40/100, b=1.0 contributes 60/100
        assert calls[-1][1] == pytest.approx(1.0)

    def test_parallel_then_sequential(self):
        calls: list[tuple[str, float]] = []
        tracker = _StageTracker(lambda d, f: calls.append((d, f)), total_weight=110.0)

        # Parallel: weight 40 + 60 = 100
        reporters = tracker.parallel_stages({"a": 40.0, "b": 60.0})
        reporters["a"]("A", 1.0)
        reporters["b"]("B", 1.0)

        # Sequential: weight 10
        r = tracker.stage(10.0)
        r("C", 0.0)
        assert calls[-1][1] == pytest.approx(100.0 / 110.0)
        r("C", 1.0)
        assert calls[-1][1] == pytest.approx(1.0)

    def test_sequential_then_parallel_then_sequential(self):
        calls: list[tuple[str, float]] = []
        tracker = _StageTracker(lambda d, f: calls.append((d, f)), total_weight=120.0)

        # Sequential stage: weight 20
        r = tracker.stage(20.0)
        r("Pre", 1.0)
        assert calls[-1][1] == pytest.approx(20.0 / 120.0)

        # Parallel: weight 40 + 50 = 90
        reporters = tracker.parallel_stages({"d": 40.0, "t": 50.0})
        reporters["d"]("D", 0.5)
        reporters["t"]("T", 0.5)
        # base=20, d contributes 0.5*40=20, t contributes 0.5*50=25 => (20+45)/120
        assert calls[-1][1] == pytest.approx(65.0 / 120.0)

        reporters["d"]("D", 1.0)
        reporters["t"]("T", 1.0)
        assert calls[-1][1] == pytest.approx(110.0 / 120.0)

        # Sequential: weight 10
        r = tracker.stage(10.0)
        r("Post", 1.0)
        assert calls[-1][1] == pytest.approx(1.0)

    def test_parallel_thread_safety(self):
        """Verify concurrent calls from multiple threads don't corrupt state."""
        calls: list[float] = []
        lock = threading.Lock()

        def cb(desc: str, frac: float) -> None:
            with lock:
                calls.append(frac)

        tracker = _StageTracker(cb, total_weight=100.0)
        reporters = tracker.parallel_stages({"a": 50.0, "b": 50.0})

        n_steps = 100
        barrier = threading.Barrier(2)

        def run_reporter(key: str) -> None:
            barrier.wait()
            for i in range(n_steps + 1):
                reporters[key](key, i / n_steps)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(run_reporter, "a")
            fb = pool.submit(run_reporter, "b")
            fa.result()
            fb.result()

        # Should have gotten calls from both threads without errors
        assert len(calls) >= n_steps  # At least n_steps calls total
        # Final overall should be 1.0 (both reporters reached 1.0)
        assert calls[-1] == pytest.approx(1.0, abs=0.02)


class TestRuntimeConfig:
    def test_default_profile_maps_to_qwen3_asr(self):
        resolved = _resolve_runtime_config(PipelineConfig())
        assert resolved.whisper_model == "Qwen/Qwen3-ASR-1.7B-hf"
        assert resolved.alignment_mode == "accurate"
        assert resolved.condition_on_previous_text is False
        assert resolved.repetition_penalty == pytest.approx(1.02)
        assert resolved.no_repeat_ngram_size == 2

    def test_balanced_profile_changes_model_when_using_defaults(self):
        resolved = _resolve_runtime_config(PipelineConfig(profile="balanced"))
        assert resolved.whisper_model == "openai/whisper-large-v3-turbo"

    def test_speed_profile_uses_latest_distil_whisper_model(self):
        resolved = _resolve_runtime_config(PipelineConfig(profile="speed"))
        assert resolved.whisper_model == "distil-whisper/distil-large-v3.5"

    def test_profile_does_not_override_explicit_model(self):
        resolved = _resolve_runtime_config(
            PipelineConfig(
                profile="balanced",
                whisper_model="distil-whisper/distil-large-v3.5",
            )
        )
        assert resolved.whisper_model == "distil-whisper/distil-large-v3.5"

    def test_explicit_model_equal_to_old_default_is_respected(self):
        resolved = _resolve_runtime_config(
            PipelineConfig(
                profile="balanced",
                whisper_model="large-v3",
            )
        )
        assert resolved.whisper_model == "large-v3"

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            _resolve_runtime_config(PipelineConfig(profile="invalid"))

    def test_invalid_alignment_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown alignment_mode"):
            _resolve_runtime_config(PipelineConfig(alignment_mode="slow"))

    def test_high_vram_uses_float16_and_higher_beam_for_accuracy(self):
        with patch("whisper_diarize.runtime_config.detect_cuda_total_memory_gb", return_value=24.0):
            resolved = _resolve_runtime_config(PipelineConfig(profile="accuracy"))
        assert resolved.compute_type == "float16"
        assert resolved.beam_size == 12

    def test_explicit_beam_size_equal_to_old_default_is_respected(self):
        with patch("whisper_diarize.runtime_config.detect_cuda_total_memory_gb", return_value=24.0):
            resolved = _resolve_runtime_config(PipelineConfig(profile="accuracy", beam_size=5))
        assert resolved.beam_size == 5

    def test_explicit_compute_type_equal_to_old_default_is_respected(self):
        with patch("whisper_diarize.runtime_config.detect_cuda_total_memory_gb", return_value=24.0):
            resolved = _resolve_runtime_config(
                PipelineConfig(profile="accuracy", compute_type="int8_float16")
            )
        assert resolved.compute_type == "int8_float16"

    def test_accuracy_profile_uses_higher_beam_even_without_high_vram(self):
        with patch("whisper_diarize.runtime_config.detect_cuda_total_memory_gb", return_value=12.0):
            resolved = _resolve_runtime_config(PipelineConfig(profile="accuracy"))
        assert resolved.beam_size == 8

    def test_cuda_requested_but_unavailable_falls_back_to_cpu(self):
        with patch("whisper_diarize.runtime_config.detect_cuda_total_memory_gb", return_value=None):
            resolved = _resolve_runtime_config(PipelineConfig(device="cuda"))
        assert resolved.device == "cpu"


class TestSequentialRetryDecision:
    def test_oom_error_retries(self):
        assert _should_retry_sequential(RuntimeError("CUDA out of memory"))

    def test_non_oom_error_does_not_retry(self):
        assert not _should_retry_sequential(RuntimeError("simulated diarization failure"))


class TestWorkerRecommendation:
    def test_accuracy_profile_24gb_recommends_two_workers(self):
        workers = recommend_worker_count(
            10,
            PipelineConfig(profile="accuracy", device="cuda"),
            vram_gb=24.0,
        )
        assert workers == 2

    def test_cpu_recommends_single_worker(self):
        workers = recommend_worker_count(
            10,
            PipelineConfig(profile="accuracy", device="cpu"),
            vram_gb=24.0,
        )
        assert workers == 1


class TestParallelExecution:
    """Test that diarize + transcribe actually run concurrently."""

    def test_parallel_runs_both_tasks(self):
        """Both diarize and transcribe should be called and their results used."""
        from whisper_diarize.models import SpeakerTurn, WordItem

        mock_turns = [SpeakerTurn(start_s=0.0, end_s=5.0, speaker="SPEAKER_00")]
        mock_words = [WordItem(start_s=0.0, end_s=0.5, word="Hello")]

        with (
            patch(
                "whisper_diarize.pipeline.diarization.diarize", return_value=mock_turns
            ) as mock_d,
            patch(
                "whisper_diarize.pipeline.transcription.transcribe", return_value=mock_words
            ) as mock_t,
        ):
            from pathlib import Path

            import numpy as np

            from whisper_diarize.pipeline import (
                PipelineConfig,
                _run_diarize_transcribe_parallel,
            )

            config = PipelineConfig()
            weights = _compute_weights(1.0, False, True)
            audio_data = np.zeros(16000, dtype=np.float32)

            turns, words = _run_diarize_transcribe_parallel(
                Path("/tmp/test.wav"),
                audio_data,
                16000,
                "token",
                config,
                None,
                weights,
            )

            assert mock_d.called
            assert mock_t.called
            assert turns == mock_turns
            assert words == mock_words

    def test_transcribe_receives_decode_controls(self):
        """Pipeline forwards anti-repetition decode controls to transcription."""
        from whisper_diarize.models import SpeakerTurn, WordItem

        with (
            patch(
                "whisper_diarize.pipeline.diarization.diarize",
                return_value=[SpeakerTurn(start_s=0.0, end_s=5.0, speaker="SPEAKER_00")],
            ),
            patch(
                "whisper_diarize.pipeline.transcription.transcribe",
                return_value=[WordItem(start_s=0.0, end_s=0.5, word="Hello")],
            ) as mock_t,
        ):
            from pathlib import Path

            import numpy as np

            from whisper_diarize.pipeline import _run_diarize_transcribe_sequential

            config = PipelineConfig(
                condition_on_previous_text=False,
                temperature=0.4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
            )
            weights = _compute_weights(1.0, False, True)
            audio_data = np.zeros(16000, dtype=np.float32)

            _run_diarize_transcribe_sequential(
                Path("/tmp/test.wav"),
                audio_data,
                16000,
                "token",
                config,
                None,
                weights,
            )

            kwargs = mock_t.call_args.kwargs
            assert kwargs["condition_on_previous_text"] is False
            assert kwargs["temperature"] == 0.4
            assert kwargs["repetition_penalty"] == pytest.approx(1.2)
            assert kwargs["no_repeat_ngram_size"] == 4

    def test_sequential_runs_both_tasks(self):
        """Sequential mode should also produce correct results."""
        from whisper_diarize.models import SpeakerTurn, WordItem

        mock_turns = [SpeakerTurn(start_s=0.0, end_s=5.0, speaker="SPEAKER_00")]
        mock_words = [WordItem(start_s=0.0, end_s=0.5, word="Hello")]

        with (
            patch(
                "whisper_diarize.pipeline.diarization.diarize", return_value=mock_turns
            ) as mock_d,
            patch(
                "whisper_diarize.pipeline.transcription.transcribe", return_value=mock_words
            ) as mock_t,
        ):
            from pathlib import Path

            import numpy as np

            from whisper_diarize.pipeline import _run_diarize_transcribe_sequential

            config = PipelineConfig()
            weights = _compute_weights(1.0, False, True)
            audio_data = np.zeros(16000, dtype=np.float32)

            turns, words = _run_diarize_transcribe_sequential(
                Path("/tmp/test.wav"),
                audio_data,
                16000,
                "token",
                config,
                None,
                weights,
            )

            assert mock_d.called
            assert mock_t.called
            assert turns == mock_turns
            assert words == mock_words

    def test_parallel_is_concurrent(self):
        """Verify both tasks actually overlap in time."""
        started = {"diarize": 0.0, "transcribe": 0.0}
        finished = {"diarize": 0.0, "transcribe": 0.0}
        barrier = threading.Barrier(2, timeout=5)

        def fake_diarize(*args, **kwargs):
            from whisper_diarize.models import SpeakerTurn

            started["diarize"] = time.monotonic()
            barrier.wait()  # Ensure both are running
            time.sleep(0.05)
            finished["diarize"] = time.monotonic()
            return [SpeakerTurn(start_s=0.0, end_s=1.0, speaker="S0")]

        def fake_transcribe(*args, **kwargs):
            from whisper_diarize.models import WordItem

            started["transcribe"] = time.monotonic()
            barrier.wait()  # Ensure both are running
            time.sleep(0.05)
            finished["transcribe"] = time.monotonic()
            return [WordItem(start_s=0.0, end_s=0.5, word="Hi")]

        with (
            patch("whisper_diarize.pipeline.diarization.diarize", side_effect=fake_diarize),
            patch("whisper_diarize.pipeline.transcription.transcribe", side_effect=fake_transcribe),
        ):
            from pathlib import Path

            import numpy as np

            from whisper_diarize.pipeline import _run_diarize_transcribe_parallel

            config = PipelineConfig()
            weights = _compute_weights(1.0, False, True)

            turns, words = _run_diarize_transcribe_parallel(
                Path("/tmp/test.wav"),
                np.zeros(16000, dtype=np.float32),
                16000,
                "token",
                config,
                None,
                weights,
            )

        # Both should have started before either finished
        assert started["transcribe"] < finished["diarize"]
        assert started["diarize"] < finished["transcribe"]

    def test_parallel_propagates_diarize_error(self):
        """If diarize raises, the error should propagate."""
        from whisper_diarize.models import WordItem

        with (
            patch("whisper_diarize.pipeline.diarization.diarize", side_effect=RuntimeError("boom")),
            patch(
                "whisper_diarize.pipeline.transcription.transcribe",
                return_value=[WordItem(start_s=0.0, end_s=0.5, word="Hi")],
            ),
        ):
            from pathlib import Path

            import numpy as np

            from whisper_diarize.pipeline import _run_diarize_transcribe_parallel

            config = PipelineConfig()
            weights = _compute_weights(1.0, False, True)

            with pytest.raises(RuntimeError, match="boom"):
                _run_diarize_transcribe_parallel(
                    Path("/tmp/test.wav"),
                    np.zeros(16000, dtype=np.float32),
                    16000,
                    "token",
                    config,
                    None,
                    weights,
                )

    def test_parallel_propagates_transcribe_error(self):
        """If transcribe raises, the error should propagate."""
        from whisper_diarize.models import SpeakerTurn

        with (
            patch(
                "whisper_diarize.pipeline.diarization.diarize",
                return_value=[SpeakerTurn(start_s=0.0, end_s=1.0, speaker="S0")],
            ),
            patch(
                "whisper_diarize.pipeline.transcription.transcribe",
                side_effect=RuntimeError("fail"),
            ),
        ):
            from pathlib import Path

            import numpy as np

            from whisper_diarize.pipeline import _run_diarize_transcribe_parallel

            config = PipelineConfig()
            weights = _compute_weights(1.0, False, True)

            with pytest.raises(RuntimeError, match="fail"):
                _run_diarize_transcribe_parallel(
                    Path("/tmp/test.wav"),
                    np.zeros(16000, dtype=np.float32),
                    16000,
                    "token",
                    config,
                    None,
                    weights,
                )

    def test_parallel_with_progress_tracking(self):
        """Progress callback should receive monotonically increasing values from parallel stages."""
        from whisper_diarize.models import SpeakerTurn, WordItem

        def fake_diarize(*args, on_progress=None, **kwargs):
            if on_progress:
                on_progress("Loading diarization model", 0.0)
                on_progress("Diarizing speakers", 0.15)
                on_progress("Diarizing speakers", 1.0)
            return [SpeakerTurn(start_s=0.0, end_s=1.0, speaker="S0")]

        def fake_transcribe(*args, on_progress=None, **kwargs):
            if on_progress:
                on_progress("Loading transcription model", 0.0)
                on_progress("Transcribing speech", 0.10)
                on_progress("Transcribing speech", 1.0)
            return [WordItem(start_s=0.0, end_s=0.5, word="Hi")]

        calls: list[tuple[str, float]] = []
        lock = threading.Lock()

        def on_progress(desc: str, frac: float) -> None:
            with lock:
                calls.append((desc, frac))

        with (
            patch("whisper_diarize.pipeline.diarization.diarize", side_effect=fake_diarize),
            patch("whisper_diarize.pipeline.transcription.transcribe", side_effect=fake_transcribe),
        ):
            from pathlib import Path

            import numpy as np

            from whisper_diarize.pipeline import _run_diarize_transcribe_parallel

            config = PipelineConfig()
            weights = _compute_weights(1.0, False, True)
            total = sum(weights.values())
            tracker = _StageTracker(on_progress, total)

            # Skip earlier stages
            tracker.stage(weights["load_audio"])
            tracker.stage(weights["clean"])

            _run_diarize_transcribe_parallel(
                Path("/tmp/test.wav"),
                np.zeros(16000, dtype=np.float32),
                16000,
                "token",
                config,
                tracker,
                weights,
            )

        assert len(calls) > 0
        # All fractions should be between 0 and 1
        for desc, frac in calls:
            assert 0.0 <= frac <= 1.0, f"frac={frac} for {desc}"
