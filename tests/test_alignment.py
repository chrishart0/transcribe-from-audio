"""Tests for alignment module."""

from __future__ import annotations

import pytest

from whisper_diarize.alignment import align_words_to_speakers, find_speaker_at_time
from whisper_diarize.models import SpeakerTurn, WordItem


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

