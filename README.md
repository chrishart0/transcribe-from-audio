# whisper-diarize

Local diarized transcription using **Qwen3-ASR**, **Qwen3-ForcedAligner**, and
**Nemotron 3 Diarization** by default. The balanced and speed profiles use
**faster-whisper** and **pyannote.audio**.

## Installation

```bash
pip install whisper-diarize
```

For development:

```bash
uv sync --all-extras
```

## Quick Start

```bash
# Basic usage
transcribe audio.mp3

# Accuracy-first profile (recommended default)
transcribe audio.mp3 --profile accuracy

# Balanced profile
transcribe audio.mp3 --hf-token "$HF_TOKEN" --profile balanced

# Speed profile
transcribe audio.mp3 --hf-token "$HF_TOKEN" --profile speed

# Process a whole folder
transcribe ./recordings/

# Recursive folder processing with output directory
transcribe ./recordings/ --recursive --output-dir ./transcripts/

# Process multiple files concurrently
transcribe ./recordings/ --workers 3

# Auto-scale workers from available VRAM (default behavior)
transcribe ./recordings/ --workers 0

# CPU-only mode
transcribe audio.mp3 --hf-token $HF_TOKEN --device cpu --compute-type int8

# Specify number of speakers (selects pyannote Community-1)
transcribe audio.mp3 --hf-token "$HF_TOKEN" --num-speakers 2
```

The default accuracy profile needs no Hugging Face token and handles up to eight
speakers. Speaker-count hints and pyannote tuning options automatically select
Community-1 unless you explicitly choose a diarization model. Community-1 requires
a Hugging Face token and acceptance of its model terms. Set `HF_TOKEN` in your
environment or copy `.env.example` to `.env` for that backend.

### Transcription Profiles

| Profile | Transcription | Diarization | Timestamps |
|---------|---------------|-------------|------------|
| `accuracy` | `Qwen/Qwen3-ASR-1.7B-hf` | `nvidia/Nemotron-3-Diarization` | Qwen3-ForcedAligner-0.6B-hf |
| `balanced` | `openai/whisper-large-v3-turbo` | pyannote Community-1 | Whisper word timestamps |
| `speed` | `distil-whisper/distil-large-v3.5` | pyannote Community-1 | Whisper word timestamps |

Accuracy defaults to English. Use `--language` for another supported language.
Forced alignment supports Chinese, English, Cantonese, French, German, Italian,
Japanese, Korean, Portuguese, Russian, and Spanish. Japanese requires `nagisa` and
Korean requires `soynlp`, installed separately with `uv add`.

The OpenAI and Distil-Whisper names resolve to CTranslate2 checkpoints that
faster-whisper can load. Use `balanced` for languages outside the forced aligner's
coverage. The speed profile supports English only. These profiles transcribe speech
in its original language; this CLI does not expose translation.

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hf-token` | none | Required for pyannote; optional for the default accuracy profile |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--profile` | `accuracy` | Preset (`accuracy`, `balanced`, `speed`) |
| `--model` | profile default | Transcription model alias or Hugging Face model ID |
| `--diarization-model` | profile default | Nemotron or pyannote model ID |
| `--compute-type` | profile default | CTranslate2 compute type (Whisper only) |
| `--beam-size` | profile default | Beam search size (Whisper only) |
| `--[no-]condition-on-previous-text` | profile default | Condition on previous text across segments |
| `--temperature` | profile default | Decoding temperature override |
| `--repetition-penalty` | profile default | Penalize repetitive decoding |
| `--no-repeat-ngram-size` | profile default | Prevent repeating n-grams (`0` disables) |
| `--language` | English for Qwen, auto for Whisper | Language code (e.g., `en`, `es`) |
| `--alignment-mode` | profile default | Alignment behavior (`accurate`, `fast`) |
| `--output-dir` | next to input | Directory for output files |
| `--recursive` | false | Search subdirectories when input is a directory |
| `--no-clean` | false | Skip audio preprocessing |
| `--save-cleaned` | false | Save cleaned audio as `.cleaned.wav` |
| `--highpass` | `80` | High-pass filter frequency (Hz) |
| `--noise-reduction` | `0.8` | Noise reduction strength (0-1) |
| `--num-speakers` | auto | Exact number of speakers |
| `--min-speakers` | none | Minimum expected speakers |
| `--max-speakers` | none | Maximum expected speakers |
| `--workers` | `0` | Number of files to process concurrently (`0` = auto based on VRAM) |
| `--no-parallel` | false | Disable parallel diarization/transcription per file |
| `--no-retry-sequential` | false | Disable retrying sequential mode after OOM in parallel mode |

### GPU Auto-Tuning

- On CUDA systems, runtime config is auto-tuned from available VRAM.
- `accuracy` uses Qwen3-ASR in BF16 on CUDA, processing four 20-second chunks at a time.
- Qwen word timestamps come from forced alignment rather than evenly spaced estimates.
- Nemotron retains short speaker replies and overlapping turns. Pyannote uses the existing turn smoothing.
- `balanced` uses Whisper large-v3-turbo. `speed` uses Distil-Whisper large-v3.5 (English-only).
- Whisper compute type is selected from available VRAM. Its beam and quantization options do not affect Qwen.
- For directory inputs, `--workers 0` auto-selects concurrency from VRAM and profile.
- Whisper anti-repetition defaults use `condition_on_previous_text=False`, repetition penalty, and n-gram blocking.

On one 180-second meeting sample, batched Qwen produced identical normalized text
in 2.0 seconds versus 5.0 seconds sequentially on an RTX 5090. Forced alignment
added 0.34 seconds. Nemotron with forced alignment agreed with the existing named
transcript on 95.6% of 458 matched tokens, versus 82.1% for the previous defaults.
These are results from one recording and an unverified reference, not general
accuracy guarantees or formal diarization error rates.

## Output Files

For an input file `recording.mp3`, the pipeline produces:

- `recording.diarized.json` - Structured data with timestamps
- `recording.diarized.txt` - Readable dialogue format
- `recording.diarized.srt` - Subtitles with speaker labels
- `recording.cleaned.wav` - Cleaned audio (if `--save-cleaned`)

## Python API

```python
from pathlib import Path
from whisper_diarize import run, PipelineConfig

config = PipelineConfig(
    profile="accuracy",
    device="cuda",
    clean_audio=True,
)

result = run(
    Path("audio.mp3"),
    config=config,
)

for utterance in result.utterances:
    print(f"{utterance.speaker}: {utterance.text}")
```

## Package Structure

```
whisper_diarize/
├── __init__.py      # Public API exports
├── models.py        # Data models (SpeakerTurn, WordItem, Utterance)
├── audio.py         # Audio loading and cleaning
├── diarization.py   # Diarization backend selection
├── nemotron.py      # Nemotron 3 speaker diarization
├── transcription.py # Speech-to-text (Qwen3-ASR or faster-whisper)
├── qwen_asr.py      # Batched Qwen3-ASR and forced alignment
├── alignment.py     # Align words to speakers
├── output.py        # Output formatters (JSON, TXT, SRT)
├── pipeline.py      # High-level orchestration
└── cli.py           # Command-line interface
```

## Requirements

- Python 3.11+
- `faster-whisper` 1.2.1+ (includes `distil-large-v3.5` support and the current VAD model)
- `transformers` 5.22.1+ for Qwen3-ASR, forced alignment, and Nemotron (installed by default)
- `pyannote.audio` 4.0+ (required by the Community-1 pipeline)
- For pyannote: a Hugging Face token with access to `pyannote/speaker-diarization-community-1`
- NVIDIA GPU recommended (CPU works but slower)

CUDA must be available to PyTorch. If CUDA initialization fails, the application
warns and falls back to CPU. An NVIDIA driver/library mismatch requires fixing the
host driver installation; model or beam settings cannot restore GPU access.

## Running Tests

```bash
uv run pytest tests/ -v
```
