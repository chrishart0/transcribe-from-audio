"""Tests for alignment module."""

from __future__ import annotations

from whisper_diarize.alignment import align_words_to_speakers, find_speaker_at_time, smooth_turns
from whisper_diarize.models import SpeakerTurn, WordItem


class TestSmoothTurns:
    def test_removes_short_turns(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="A"),
            SpeakerTurn(start_s=5.0, end_s=5.3, speaker="B"),  # 0.3s — too short
            SpeakerTurn(start_s=5.3, end_s=10.0, speaker="A"),
        ]
        result = smooth_turns(turns, min_turn_s=0.5, merge_gap_s=0.0, flicker_s=0.0)
        assert len(result) == 2
        assert all(t.speaker == "A" for t in result)

    def test_merges_same_speaker_gaps(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="A"),
            SpeakerTurn(start_s=5.2, end_s=10.0, speaker="A"),  # 0.2s gap
        ]
        result = smooth_turns(turns, min_turn_s=0.0, merge_gap_s=0.3, flicker_s=0.0)
        assert len(result) == 1
        assert result[0].start_s == 0.0
        assert result[0].end_s == 10.0
        assert result[0].speaker == "A"

    def test_does_not_merge_different_speakers(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="A"),
            SpeakerTurn(start_s=5.1, end_s=10.0, speaker="B"),
        ]
        result = smooth_turns(turns, min_turn_s=0.0, merge_gap_s=0.3, flicker_s=0.0)
        assert len(result) == 2

    def test_does_not_merge_large_gap(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="A"),
            SpeakerTurn(start_s=5.5, end_s=10.0, speaker="A"),  # 0.5s gap > 0.3 threshold
        ]
        result = smooth_turns(turns, min_turn_s=0.0, merge_gap_s=0.3, flicker_s=0.0)
        assert len(result) == 2

    def test_fixes_flicker(self):
        # A→B→A where B is short (0.5s < 1.0s flicker threshold)
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="A"),
            SpeakerTurn(start_s=5.0, end_s=5.5, speaker="B"),  # flicker
            SpeakerTurn(start_s=5.5, end_s=10.0, speaker="A"),
        ]
        result = smooth_turns(turns, min_turn_s=0.0, merge_gap_s=0.0, flicker_s=1.0)
        assert len(result) == 1
        assert result[0].speaker == "A"
        assert result[0].start_s == 0.0
        assert result[0].end_s == 10.0

    def test_does_not_fix_long_flicker(self):
        # B turn is 1.5s which exceeds flicker_s=1.0
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="A"),
            SpeakerTurn(start_s=5.0, end_s=6.5, speaker="B"),
            SpeakerTurn(start_s=6.5, end_s=10.0, speaker="A"),
        ]
        result = smooth_turns(turns, min_turn_s=0.0, merge_gap_s=0.0, flicker_s=1.0)
        assert len(result) == 3

    def test_empty_turns(self):
        assert smooth_turns([]) == []

    def test_zero_thresholds_is_noop(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=0.1, speaker="A"),
            SpeakerTurn(start_s=0.1, end_s=0.2, speaker="B"),
            SpeakerTurn(start_s=0.2, end_s=0.3, speaker="A"),
        ]
        result = smooth_turns(turns, min_turn_s=0.0, merge_gap_s=0.0, flicker_s=0.0)
        assert len(result) == 3

    def test_all_passes_combined(self):
        # Short B turn gets removed, then A turns get merged
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="A"),
            SpeakerTurn(start_s=5.0, end_s=5.2, speaker="B"),  # short: removed
            SpeakerTurn(start_s=5.2, end_s=10.0, speaker="A"),
        ]
        result = smooth_turns(turns, min_turn_s=0.5, merge_gap_s=0.3, flicker_s=1.0)
        assert len(result) == 1
        assert result[0].speaker == "A"


class TestFindSpeakerAtTime:
    def test_finds_speaker_in_turn(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="SPEAKER_00"),
            SpeakerTurn(start_s=5.5, end_s=10.0, speaker="SPEAKER_01"),
        ]
        assert find_speaker_at_time(turns, 2.5) == "SPEAKER_00"
        assert find_speaker_at_time(turns, 7.0) == "SPEAKER_01"

    def test_returns_none_for_large_gap(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="SPEAKER_00"),
            SpeakerTurn(start_s=10.0, end_s=15.0, speaker="SPEAKER_01"),
        ]
        assert find_speaker_at_time(turns, 7.5, gap_tolerance=0.5) is None

    def test_returns_nearest_within_tolerance(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=5.0, speaker="SPEAKER_00"),
            SpeakerTurn(start_s=5.4, end_s=10.0, speaker="SPEAKER_01"),
        ]
        # 5.1 is 0.1s from SPEAKER_00's end (5.0) and 0.3s from SPEAKER_01's start (5.4)
        assert find_speaker_at_time(turns, 5.1, gap_tolerance=0.5) == "SPEAKER_00"

    def test_returns_none_for_empty_turns(self):
        assert find_speaker_at_time([], 1.0) is None

    def test_boundary_inclusive(self):
        turns = [SpeakerTurn(start_s=1.0, end_s=2.0, speaker="A")]
        assert find_speaker_at_time(turns, 1.0) == "A"
        assert find_speaker_at_time(turns, 2.0) == "A"


class TestAlignWordsToSpeakers:
    def test_single_speaker_continuous(self):
        turns = [SpeakerTurn(start_s=0.0, end_s=10.0, speaker="SPEAKER_00")]
        words = [
            WordItem(start_s=0.0, end_s=0.5, word="Hello"),
            WordItem(start_s=0.6, end_s=1.0, word="world"),
        ]
        utterances = align_words_to_speakers(turns, words)
        assert len(utterances) == 1
        assert utterances[0].speaker == "SPEAKER_00"
        assert utterances[0].text == "Hello world"

    def test_speaker_change_splits_utterance(self):
        turns = [
            SpeakerTurn(start_s=0.0, end_s=2.0, speaker="SPEAKER_00"),
            SpeakerTurn(start_s=2.0, end_s=4.0, speaker="SPEAKER_01"),
        ]
        words = [
            WordItem(start_s=0.5, end_s=1.0, word="Hi"),
            WordItem(start_s=2.5, end_s=3.0, word="Hey"),
        ]
        utterances = align_words_to_speakers(turns, words)
        assert len(utterances) == 2
        assert utterances[0].speaker == "SPEAKER_00"
        assert utterances[0].text == "Hi"
        assert utterances[1].speaker == "SPEAKER_01"
        assert utterances[1].text == "Hey"

    def test_gap_splits_utterance(self):
        turns = [SpeakerTurn(start_s=0.0, end_s=10.0, speaker="SPEAKER_00")]
        words = [
            WordItem(start_s=0.0, end_s=0.5, word="Hello"),
            WordItem(start_s=2.0, end_s=2.5, word="World"),
        ]
        utterances = align_words_to_speakers(turns, words, max_gap_s=0.9)
        assert len(utterances) == 2

    def test_empty_words(self):
        turns = [SpeakerTurn(start_s=0.0, end_s=10.0, speaker="SPEAKER_00")]
        utterances = align_words_to_speakers(turns, [])
        assert utterances == []

    def test_unknown_speaker_for_large_gap(self):
        turns = [SpeakerTurn(start_s=5.0, end_s=10.0, speaker="SPEAKER_00")]
        words = [WordItem(start_s=0.0, end_s=0.5, word="Hello")]
        utterances = align_words_to_speakers(turns, words, gap_tolerance=0.5)
        assert len(utterances) == 1
        assert utterances[0].speaker == "UNKNOWN"
