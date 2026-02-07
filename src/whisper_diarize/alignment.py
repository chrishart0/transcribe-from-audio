"""Align transcribed words with speaker diarization."""

from __future__ import annotations

from whisper_diarize.models import SpeakerTurn, Utterance, WordItem


def smooth_turns(
    turns: list[SpeakerTurn],
    min_turn_s: float = 0.5,
    merge_gap_s: float = 0.3,
    flicker_s: float = 1.0,
) -> list[SpeakerTurn]:
    """
    Clean up noisy diarization turns before alignment.

    Three passes (each skipped when its threshold is 0.0):
    1. Remove turns shorter than min_turn_s
    2. Merge adjacent same-speaker turns separated by < merge_gap_s
    3. Fix flickers: if A→B→A and B's duration < flicker_s, reassign B to A, then re-merge
    """
    # Pass 1: remove short turns
    if min_turn_s > 0.0:
        turns = [t for t in turns if (t.end_s - t.start_s) >= min_turn_s]

    # Pass 2: merge same-speaker gaps
    if merge_gap_s > 0.0:
        turns = _merge_same_speaker(turns, merge_gap_s)

    # Pass 3: fix flickers (A→B→A where B is short → reassign B to A)
    if flicker_s > 0.0:
        merged = []
        i = 0
        while i < len(turns):
            if (
                i + 2 < len(turns)
                and turns[i].speaker == turns[i + 2].speaker
                and turns[i].speaker != turns[i + 1].speaker
                and (turns[i + 1].end_s - turns[i + 1].start_s) < flicker_s
            ):
                # Reassign middle turn to surrounding speaker
                merged.append(turns[i])
                merged.append(
                    SpeakerTurn(
                        start_s=turns[i + 1].start_s,
                        end_s=turns[i + 1].end_s,
                        speaker=turns[i].speaker,
                    )
                )
                merged.append(turns[i + 2])
                i += 3
            else:
                merged.append(turns[i])
                i += 1
        # Re-merge after flicker fix
        turns = _merge_same_speaker(merged, merge_gap_s=float("inf"))

    return turns


def _merge_same_speaker(turns: list[SpeakerTurn], merge_gap_s: float) -> list[SpeakerTurn]:
    """Merge adjacent same-speaker turns separated by less than merge_gap_s."""
    if not turns:
        return []
    merged = [turns[0]]
    for t in turns[1:]:
        prev = merged[-1]
        if t.speaker == prev.speaker and (t.start_s - prev.end_s) < merge_gap_s:
            merged[-1] = SpeakerTurn(start_s=prev.start_s, end_s=t.end_s, speaker=prev.speaker)
        else:
            merged.append(t)
    return merged


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
