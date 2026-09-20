"""Runtime configuration resolution and hardware-aware defaults."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisper_diarize.pipeline import PipelineConfig

_HIGH_VRAM_THRESHOLD_GB = 20.0
_LOW_VRAM_THRESHOLD_GB = 10.0


@dataclass(frozen=True)
class _ProfileDefaults:
    whisper_model: str
    compute_type: str
    beam_size: int
    alignment_mode: str
    vad_filter: bool
    condition_on_previous_text: bool
    repetition_penalty: float
    no_repeat_ngram_size: int


_PROFILE_DEFAULTS: dict[str, _ProfileDefaults] = {
    "accuracy": _ProfileDefaults(
        whisper_model="Qwen/Qwen3-ASR-1.7B-hf",
        compute_type="int8_float16",
        beam_size=8,
        alignment_mode="accurate",
        vad_filter=True,
        condition_on_previous_text=False,
        repetition_penalty=1.02,
        no_repeat_ngram_size=2,
    ),
    "balanced": _ProfileDefaults(
        whisper_model="openai/whisper-large-v3-turbo",
        compute_type="int8_float16",
        beam_size=5,
        alignment_mode="accurate",
        vad_filter=True,
        condition_on_previous_text=False,
        repetition_penalty=1.03,
        no_repeat_ngram_size=2,
    ),
    "speed": _ProfileDefaults(
        whisper_model="distil-whisper/distil-large-v3.5",
        compute_type="int8_float16",
        beam_size=1,
        alignment_mode="fast",
        vad_filter=True,
        condition_on_previous_text=False,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    ),
}


def detect_cuda_total_memory_gb() -> float | None:
    """Return total VRAM in GB for CUDA device 0, or None when unavailable."""
    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return None
    return float(total_bytes) / (1024**3)


def recommend_worker_count(file_count: int, config: PipelineConfig, vram_gb: float | None) -> int:
    """Recommend directory worker count from profile and available VRAM."""
    if file_count <= 1:
        return 1
    if config.device != "cuda" or vram_gb is None:
        return 1

    profile = config.profile.lower()
    if profile == "accuracy":
        per_worker_gb = 10.0
    elif profile == "balanced":
        per_worker_gb = 8.0
    else:
        per_worker_gb = 6.0

    workers = int(vram_gb // per_worker_gb)
    return max(1, min(file_count, workers))


def resolve_runtime_config(config: PipelineConfig) -> PipelineConfig:
    """Apply profile defaults while preserving explicit user overrides."""
    if config.device not in {"cuda", "cpu"}:
        raise ValueError(f"Unknown device '{config.device}'. Expected 'cuda' or 'cpu'.")

    profile = config.profile.lower()
    if profile not in _PROFILE_DEFAULTS:
        valid = ", ".join(sorted(_PROFILE_DEFAULTS))
        raise ValueError(f"Unknown profile '{config.profile}'. Expected one of: {valid}")

    defaults = _PROFILE_DEFAULTS[profile]
    compute_type_user_set = config.compute_type is not None
    beam_size_user_set = config.beam_size is not None

    resolved = replace(
        config,
        profile=profile,
        whisper_model=config.whisper_model or defaults.whisper_model,
        compute_type=config.compute_type or defaults.compute_type,
        beam_size=config.beam_size if config.beam_size is not None else defaults.beam_size,
        alignment_mode=config.alignment_mode or defaults.alignment_mode,
        vad_filter=config.vad_filter if config.vad_filter is not None else defaults.vad_filter,
        condition_on_previous_text=(
            config.condition_on_previous_text
            if config.condition_on_previous_text is not None
            else defaults.condition_on_previous_text
        ),
        repetition_penalty=(
            config.repetition_penalty
            if config.repetition_penalty is not None
            else defaults.repetition_penalty
        ),
        no_repeat_ngram_size=(
            config.no_repeat_ngram_size
            if config.no_repeat_ngram_size is not None
            else defaults.no_repeat_ngram_size
        ),
    )

    if resolved.alignment_mode is None:
        raise ValueError("alignment_mode could not be resolved")
    alignment_mode = resolved.alignment_mode.lower()
    if alignment_mode not in {"accurate", "fast"}:
        raise ValueError(
            f"Unknown alignment_mode '{resolved.alignment_mode}'. Expected 'accurate' or 'fast'."
        )
    resolved = replace(resolved, alignment_mode=alignment_mode)

    if resolved.device == "cuda":
        vram_gb = detect_cuda_total_memory_gb()
        if vram_gb is None:
            warnings.warn(
                "CUDA requested but unavailable; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            resolved = replace(resolved, device="cpu")
        else:
            updates: dict[str, object] = {}
            if not compute_type_user_set:
                if vram_gb >= _HIGH_VRAM_THRESHOLD_GB:
                    updates["compute_type"] = "float16"
                elif vram_gb < _LOW_VRAM_THRESHOLD_GB:
                    updates["compute_type"] = "int8"
            if (
                profile == "accuracy"
                and not beam_size_user_set
                and vram_gb >= _HIGH_VRAM_THRESHOLD_GB
            ):
                updates["beam_size"] = 12
            resolved = replace(resolved, **updates) if updates else resolved
    elif config.device == "cpu" and not compute_type_user_set:
        resolved = replace(resolved, compute_type="int8")

    return replace(resolved, _resolved=True)
