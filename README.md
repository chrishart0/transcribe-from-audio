# whisper-diarize

Local diarized transcription using **faster-whisper** (Whisper-family models via CTranslate2) and **pyannote.audio** for speaker diarization.

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
# Copy and configure environment
cp .env.example .env
# Edit .env with your Hugging Face token

# Basic usage
transcribe audio.mp3 --hf-token YOUR_HUGGINGFACE_TOKEN

# Accuracy-first profile (recommended default)
transcribe audio.mp3 --hf-token $HF_TOKEN --profile accuracy

# Balanced profile
transcribe audio.mp3 --hf-token $HF_TOKEN --profile balanced

# Speed profile
transcribe audio.mp3 --hf-token $HF_TOKEN --profile speed

# Process a whole folder
transcribe ./recordings/ --hf-token $HF_TOKEN

# Recursive folder processing with output directory
transcribe ./recordings/ --recursive --output-dir ./transcripts/

# Process multiple files concurrently
transcribe ./recordings/ --workers 3

# Auto-scale workers from available VRAM (default behavior)
transcribe ./recordings/ --workers 0

# CPU-only mode
transcribe audio.mp3 --hf-token $HF_TOKEN --device cpu --compute-type int8

# Specify number of speakers
transcribe audio.mp3 --hf-token $HF_TOKEN --num-speakers 2
```

### Transcription Profiles

| Profile | Default model | Language coverage | Tradeoff |
|---------|---------------|-------------------|----------|
| `accuracy` | `openai/whisper-large-v3` | Multilingual | Best accuracy; highest VRAM use |
| `balanced` | `openai/whisper-large-v3-turbo` | Multilingual transcription | Faster and lighter with a small accuracy tradeoff |
| `speed` | `distil-whisper/distil-large-v3.5` | English only | Fastest default; slightly weaker on long-form audio than turbo |

The OpenAI and Distil-Whisper names above are resolved to CTranslate2 checkpoints that
faster-whisper can load. Use `balanced` instead of `speed` for non-English audio. Whisper Turbo
is intended for transcription; use `accuracy` if speech-to-English translation quality matters.

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hf-token` | required | Hugging Face token for pyannote models |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--profile` | `accuracy` | Preset (`accuracy`, `balanced`, `speed`) |
| `--model` | profile default | Whisper model alias or Hugging Face model ID |
| `--compute-type` | profile default | CTranslate2 compute type |
| `--beam-size` | profile default | Beam search size |
| `--[no-]condition-on-previous-text` | profile default | Condition on previous text across segments |
| `--temperature` | profile default | Decoding temperature override |
| `--repetition-penalty` | profile default | Penalize repetitive decoding |
| `--no-repeat-ngram-size` | profile default | Prevent repeating n-grams (`0` disables) |
| `--language` | auto | Language code (e.g., `en`, `es`) |
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
- `accuracy` profile now favors higher-quality decoding by default (larger beam, less restrictive anti-repeat penalties).
- With high VRAM (for example 24GB), `accuracy` profile automatically prefers more aggressive settings (e.g. `float16`, even larger beam size).
- For directory inputs, `--workers 0` auto-selects concurrency from VRAM and profile.
- Anti-repetition defaults are tuned for long-form audio (`condition_on_previous_text=False`, repetition penalty, n-gram blocking).

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
    num_speakers=2,
)

result = run(
    Path("audio.mp3"),
    hf_token="hf_...",
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
├── diarization.py   # Speaker diarization (pyannote)
├── transcription.py # Speech-to-text (faster-whisper)
├── alignment.py     # Align words to speakers
├── output.py        # Output formatters (JSON, TXT, SRT)
├── pipeline.py      # High-level orchestration
└── cli.py           # Command-line interface
```

## Requirements

- Python 3.10+
- `faster-whisper` 1.2.1+ (includes `distil-large-v3.5` support and the current VAD model)
- `pyannote.audio` 4.0+ (required by the Community-1 pipeline)
- Hugging Face token with access to:
  - `pyannote/speaker-diarization-community-1`
- NVIDIA GPU recommended (CPU works but slower)

## Running Tests

```bash
uv run pytest tests/ -v
```
