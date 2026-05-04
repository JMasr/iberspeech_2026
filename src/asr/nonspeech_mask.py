"""AudioSet non-speech mask: drop ASR tokens whose midpoint falls inside a
laughter / breath / cough / applause / animal / vehicle region.

Eval rule compliance: the organizers banned these from transcriptions. We
combine three sources of "non-speech":
  - The Whisper-internal special tokens (we already strip these in normalize).
  - The dialect-specific lexicon ``[risas]``, ``[susp]``, etc. (handled in normalize).
  - The Stage 0B BEATs intervals (handled here).

Input is a list of (word, start_s, end_s, conf) tuples from the ASR pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordSpan:
    word: str
    start_s: float
    end_s: float
    confidence: float = 1.0


def apply_mask(
    words: list[WordSpan],
    nonspeech_intervals: list[tuple[float, float, str]],
) -> list[WordSpan]:
    """Drop words whose midpoint is inside any non-speech interval.

    ``nonspeech_intervals`` is a list of (start_s, end_s, tag) — exactly the
    output of ``src.data.stage0_beats.nonspeech_intervals``.
    """
    if not nonspeech_intervals:
        return list(words)

    intervals = sorted(nonspeech_intervals, key=lambda i: i[0])

    out = []
    for w in words:
        mid = 0.5 * (w.start_s + w.end_s)
        if not _midpoint_in_any(mid, intervals):
            out.append(w)
    return out


def _midpoint_in_any(mid: float, intervals: list[tuple[float, float, str]]) -> bool:
    # Binary search would help on huge lists, but COSER intervals are sparse;
    # linear scan is fine and obviously correct.
    for s, e, _tag in intervals:
        if s <= mid < e:
            return True
        if s > mid:
            break
    return False
