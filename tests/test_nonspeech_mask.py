"""Deterministic tests for src.asr.nonspeech_mask."""

from __future__ import annotations

from src.asr.nonspeech_mask import WordSpan, apply_mask


def test_apply_mask_drops_word_inside_interval():
    words = [
        WordSpan("hola", 0.0, 0.5),
        WordSpan("[risas]", 0.6, 1.2),
        WordSpan("mundo", 1.3, 1.8),
    ]
    intervals = [(0.6, 1.2, "laughter")]
    out = apply_mask(words, intervals)
    assert [w.word for w in out] == ["hola", "mundo"]


def test_apply_mask_keeps_word_outside_intervals():
    words = [WordSpan("hola", 0.0, 0.5), WordSpan("mundo", 0.6, 1.0)]
    out = apply_mask(words, [(2.0, 3.0, "cough")])
    assert [w.word for w in out] == ["hola", "mundo"]


def test_apply_mask_no_intervals_passthrough():
    words = [WordSpan("hola", 0.0, 0.5)]
    assert apply_mask(words, []) == words


def test_apply_mask_uses_midpoint_not_overlap():
    # A word that overlaps an interval but whose midpoint is outside is KEPT.
    words = [WordSpan("largo", 0.0, 2.0)]  # midpoint = 1.0
    intervals = [(1.5, 2.5, "vehicle")]
    out = apply_mask(words, intervals)
    assert len(out) == 1
