"""Output formatters for diarized transcripts."""

from __future__ import annotations

import json
from pathlib import Path

from whisper_diarize.models import Utterance


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    ms_total = int(round(seconds * 1000.0))
    hh = ms_total // 3_600_000
    mm = (ms_total % 3_600_000) // 60_000
    ss = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def to_json(utterances: list[Utterance]) -> str:
    """Convert utterances to JSON string."""
    data = [
        {
            "start": u.start_s,
            "end": u.end_s,
            "speaker": u.speaker,
            "text": u.text,
        }
        for u in utterances
    ]
    return json.dumps(data, indent=2)


def to_text(utterances: list[Utterance]) -> str:
    """Convert utterances to readable text format."""
    lines = [f"{u.speaker}: {u.text}" for u in utterances]
    return "\n".join(lines) + "\n"


def to_srt(utterances: list[Utterance]) -> str:
    """Convert utterances to SRT subtitle format."""
    lines: list[str] = []
    for i, u in enumerate(utterances, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(u.start_s)} --> {format_srt_timestamp(u.end_s)}")
        lines.append(f"{u.speaker}: {u.text}")
        lines.append("")
    return "\n".join(lines)


def write_all(out_base: Path, utterances: list[Utterance]) -> dict[str, Path]:
    """
    Write all output formats to files.

    Args:
        out_base: Base path (without extension)
        utterances: List of utterances to write

    Returns:
        Dict mapping format name to output path
    """
    outputs = {}

    json_path = Path(str(out_base) + ".diarized.json")
    json_path.write_text(to_json(utterances), encoding="utf-8")
    outputs["json"] = json_path

    txt_path = Path(str(out_base) + ".diarized.txt")
    txt_path.write_text(to_text(utterances), encoding="utf-8")
    outputs["txt"] = txt_path

    srt_path = Path(str(out_base) + ".diarized.srt")
    srt_path.write_text(to_srt(utterances), encoding="utf-8")
    outputs["srt"] = srt_path

    return outputs
