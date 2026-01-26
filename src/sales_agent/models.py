from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerTurn:
    start_s: float
    end_s: float
    speaker: str


@dataclass(frozen=True)
class WordItem:
    start_s: float
    end_s: float
    word: str


@dataclass
class Utterance:
    start_s: float
    end_s: float
    speaker: str
    text: str

