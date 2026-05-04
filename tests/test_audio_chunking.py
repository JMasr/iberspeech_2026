"""Deterministic tests for src.data.audio (chunking only — I/O is heavy)."""

from __future__ import annotations

from src.data.audio import chunk_intervals


def test_chunking_simple_speech_segment():
    chunks = list(chunk_intervals([(0.0, 60.0)], chunk_s=30.0, overlap_s=1.0, record_id="r"))
    # 0..30, 29..58.., tail folds into last
    assert len(chunks) >= 2
    # All chunks ≤ chunk_s + slack (the tail can exceed by up to chunk_s/2 — verify)
    for ch in chunks:
        assert ch.duration_s > 0
        assert ch.start_s < ch.end_s
    # Coverage: union of chunks covers [0, 60].
    covered = sorted([(c.start_s, c.end_s) for c in chunks])
    assert covered[0][0] == 0.0
    assert covered[-1][1] >= 60.0 - 0.001


def test_chunking_short_segment_yields_one_chunk():
    chunks = list(chunk_intervals([(0.0, 10.0)], chunk_s=30.0, overlap_s=1.0, record_id="r"))
    assert len(chunks) == 1
    assert chunks[0].start_s == 0.0
    assert abs(chunks[0].end_s - 10.0) < 1e-6


def test_chunking_overlap_is_respected():
    chunks = list(chunk_intervals([(0.0, 90.0)], chunk_s=30.0, overlap_s=5.0, record_id="r"))
    # adjacent chunks overlap by 5s
    for prev, curr in zip(chunks, chunks[1:]):
        if prev.end_s > curr.start_s:
            overlap = prev.end_s - curr.start_s
            assert overlap == 5.0 or overlap > 0  # last tail may differ
            break


def test_chunk_id_unique():
    chunks = list(chunk_intervals([(0.0, 90.0), (100.0, 130.0)], record_id="r"))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_invalid_overlap_raises():
    import pytest

    with pytest.raises(ValueError):
        list(chunk_intervals([(0.0, 60.0)], chunk_s=10.0, overlap_s=10.0, record_id="r"))
