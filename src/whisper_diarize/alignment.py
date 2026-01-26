"""Align transcribed words with speaker diarization."""

from __future__ import annotations

from whisper_diarize.models import SpeakerTurn, Utterance, WordItem


def find_speaker_at_time(
    turns: list[SpeakerTurn],
    t: float,
    gap_tolerance: float = 0.5,
) -> str | None:
    """
    Find which speaker is talking at a given time.

    Args:
        turns: List of speaker turns
        t: Time in seconds
        gap_tolerance: Max distance to nearest turn to still assign a speaker
    """
    # Exact match within a turn
    for turn in turns:
        if turn.start_s <= t <= turn.end_s:
            return turn.speaker

    # Find nearest turn if within tolerance
    best_speaker = None
    best_distance = float("inf")

    for turn in turns:
        if t < turn.start_s:
            distance = turn.start_s - t
        elif t > turn.end_s:
            distance = t - turn.end_s
        else:
            distance = 0

        if distance < best_distance:
            best_distance = distance
            best_speaker = turn.speaker

    if best_distance <= gap_tolerance:
        return best_speaker

    return None


def align_words_to_speakers(
    turns: list[SpeakerTurn],
    words: list[WordItem],
    max_gap_s: float = 0.9,
    gap_tolerance: float = 0.5,
) -> list[Utterance]:
    """
    Merge words into utterances based on speaker turns.

    Args:
        turns: Speaker diarization turns
        words: Transcribed words with timestamps
        max_gap_s: Max gap before starting new utterance (same speaker)
        gap_tolerance: Max distance to assign word to nearest speaker
    """
    utterances: list[Utterance] = []
    current: Utterance | None = None

    for w in words:
        midpoint = (w.start_s + w.end_s) / 2.0
        speaker = find_speaker_at_time(turns, midpoint, gap_tolerance) or "UNKNOWN"

        if current is None:
            current = Utterance(
                start_s=w.start_s,
                end_s=w.end_s,
                speaker=speaker,
                text=w.word.strip(),
            )
            continue

        gap = w.start_s - current.end_s
        speaker_changed = speaker != current.speaker

        if speaker_changed or gap > max_gap_s:
            utterances.append(current)
            current = Utterance(
                start_s=w.start_s,
                end_s=w.end_s,
                speaker=speaker,
                text=w.word.strip(),
            )
        else:
            current.end_s = w.end_s
            current.text = (current.text + " " + w.word.strip()).strip()

    if current is not None:
        utterances.append(current)

    return utterances

