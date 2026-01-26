# Sales Agent - Diarized Transcription

Local diarized transcription using **faster-whisper** (Whisper large-v3 via CTranslate2) and **pyannote.audio** for speaker diarization.

## Installation

```bash
uv sync --all-extras
```

## Quick Start

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your Hugging Face token

# Basic usage
uv run transcribe audio.mp3 --hf-token YOUR_HUGGINGFACE_TOKEN

# With audio cleanup and save cleaned version
uv run transcribe audio.mp3 --hf-token $HF_TOKEN --save-cleaned

# CPU-only mode
uv run transcribe audio.mp3 --hf-token $HF_TOKEN --device cpu --compute-type int8

# Specify number of speakers
uv run transcribe audio.mp3 --hf-token $HF_TOKEN --num-speakers 2
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--hf-token` | required | Hugging Face token for pyannote models |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--model` | `large-v3` | Whisper model size |
| `--compute-type` | `int8_float16` | CTranslate2 compute type |
| `--language` | auto | Language code (e.g., `en`, `es`) |
| `--no-clean` | false | Skip audio preprocessing |
| `--save-cleaned` | false | Save cleaned audio as `.cleaned.wav` |
| `--highpass` | `80` | High-pass filter frequency (Hz) |
| `--noise-reduction` | `0.8` | Noise reduction strength (0-1) |
| `--num-speakers` | auto | Exact number of speakers |
| `--min-speakers` | none | Minimum expected speakers |
| `--max-speakers` | none | Maximum expected speakers |

## Output Files

For an input file `recording.mp3`, the pipeline produces:

- `recording.diarized.json` - Structured data with timestamps
- `recording.diarized.txt` - Readable dialogue format
- `recording.diarized.srt` - Subtitles with speaker labels
- `recording.cleaned.wav` - Cleaned audio (if `--save-cleaned`)

## Python API

```python
from pathlib import Path
from sales_agent import run, PipelineConfig

config = PipelineConfig(
    whisper_model="large-v3",
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
src/sales_agent/
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
- Hugging Face token with access to:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`
- NVIDIA GPU recommended (CPU works but slower)

## Running Tests

```bash
uv run pytest tests/ -v
```
