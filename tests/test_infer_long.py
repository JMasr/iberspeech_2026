"""Deterministic tests for src.asr.infer_long merge logic."""

from __future__ import annotations

from src.asr.infer_long import (
    ChunkRecognition,
    chunks_excluding_music,
    merge_overlapping,
)
from src.asr.nonspeech_mask import WordSpan


def _ws(word: str, start: float, end: float) -> WordSpan:
    return WordSpan(word=word, start_s=start, end_s=end)


def test_merge_two_chunks_no_overlap():
    a = ChunkRecognition("a", 0.0, 10.0, [_ws("hola", 1.0, 1.5)])
    b = ChunkRecognition("b", 10.0, 20.0, [_ws("mundo", 11.0, 11.5)])
    out = merge_overlapping([a, b])
    assert [w.word for w in out] == ["hola", "mundo"]


def test_merge_with_overlap_consensus_kept():
    # Both chunks see "hola" near 9.5s in the overlap.
    a = ChunkRecognition("a", 0.0, 10.0, [_ws("hola", 9.4, 9.7)])
    b = ChunkRecognition("b", 9.0, 19.0, [_ws("hola", 9.4, 9.7), _ws("mundo", 11.0, 11.5)])
    out = merge_overlapping([a, b])
    assert [w.word for w in out] == ["hola", "mundo"]


def test_merge_disagreement_uses_closer_chunk_center():
    # Chunk A center is 5s, chunk B center is 14s. The disputed word at t=9.4s
    # is closer to A's center; A's word should be kept.
    a = ChunkRecognition("a", 0.0, 10.0, [_ws("hola", 9.3, 9.7)])
    b = ChunkRecognition("b", 9.0, 19.0, [_ws("ola", 9.3, 9.7)])
    out = merge_overlapping([a, b])
    assert any(w.word == "hola" for w in out)


def test_chunks_excluding_music_drops_chunks_centered_in_music():
    """Midpoint-based check: a chunk is dropped iff its midpoint is in a music interval."""
    rows = [
        {"start_s": 0.0, "end_s": 30.0},  # midpoint 15.0  → outside (40, 60)
        {"start_s": 30.0, "end_s": 60.0},  # midpoint 45.0  → INSIDE (40, 60)
        {"start_s": 60.0, "end_s": 90.0},  # midpoint 75.0  → outside
    ]
    out = chunks_excluding_music(rows, [(40.0, 60.0)])
    assert len(out) == 2
    assert {row["start_s"] for row in out} == {0.0, 60.0}


def test_chunks_excluding_music_tolerates_tiny_boundary_bleed():
    """A chunk with a 20 ms music sliver at one edge is NOT dropped (midpoint is in speech)."""
    rows = [{"start_s": 49.98, "end_s": 59.98}]  # midpoint 54.98, music ends at 50
    out = chunks_excluding_music(rows, [(30.0, 50.0)])
    assert len(out) == 1


def test_chunks_excluding_music_no_intervals_passthrough():
    rows = [{"start_s": 0.0, "end_s": 30.0}]
    assert chunks_excluding_music(rows, []) == rows
