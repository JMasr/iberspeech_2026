"""Deterministic tests for src.sd.refine (boundary snap + RTTM cleanup)."""

from __future__ import annotations

from src.sd.refine import (
    MERGE_GAP_S,
    MIN_SEGMENT_S,
    Segment,
    cleanup,
    parse_rttm,
    snap_boundaries,
    write_rttm,
)


def test_snap_to_nearest_word_edge():
    segs = [Segment(1.05, 3.10, "S0")]
    edges = [1.00, 2.00, 3.00]
    out = snap_boundaries(segs, edges, radius_s=0.20)
    assert out[0].start_s == 1.00
    assert out[0].end_s == 3.00


def test_snap_outside_radius_keeps_original():
    segs = [Segment(1.50, 3.50, "S0")]
    edges = [1.00, 4.00]
    out = snap_boundaries(segs, edges, radius_s=0.20)
    assert out[0].start_s == 1.50
    assert out[0].end_s == 3.50


def test_snap_no_edges_passthrough():
    segs = [Segment(1.0, 2.0, "A")]
    assert snap_boundaries(segs, []) == segs


def test_cleanup_drops_short_segments():
    segs = [Segment(0.0, 0.10, "A"), Segment(0.5, 1.5, "B")]
    out = cleanup(segs, file_duration_s=2.0)
    assert len(out) == 1
    assert out[0].speaker == "B"


def test_cleanup_merges_same_speaker_short_gap():
    gap = MERGE_GAP_S / 2
    segs = [Segment(0.0, 1.0, "A"), Segment(1.0 + gap, 2.0, "A")]
    out = cleanup(segs, file_duration_s=3.0)
    assert len(out) == 1
    assert out[0].start_s == 0.0
    assert out[0].end_s == 2.0


def test_cleanup_does_not_merge_across_speakers():
    gap = MERGE_GAP_S / 2
    segs = [Segment(0.0, 1.0, "A"), Segment(1.0 + gap, 2.0, "B")]
    out = cleanup(segs, file_duration_s=3.0)
    assert len(out) == 2


def test_cleanup_clamps_to_file_duration():
    segs = [Segment(1.0, 5.0, "A")]
    out = cleanup(segs, file_duration_s=3.0)
    assert len(out) == 1
    assert out[0].end_s == 3.0


def test_cleanup_drops_empty_after_clamp():
    segs = [Segment(5.0, 6.0, "A")]
    out = cleanup(segs, file_duration_s=3.0)
    assert out == []


def test_min_segment_constants_match_plan():
    assert MIN_SEGMENT_S == 0.20
    assert MERGE_GAP_S == 0.50


def test_rttm_round_trip(tmp_path):
    segs = [Segment(0.5, 1.5, "SPEAKER_00"), Segment(2.0, 3.5, "SPEAKER_01")]
    path = tmp_path / "out.rttm"
    write_rttm("rec1", segs, path)
    parsed = parse_rttm(path)
    assert len(parsed) == 2
    assert abs(parsed[0].start_s - 0.5) < 1e-6
    assert abs(parsed[0].end_s - 1.5) < 1e-6
    assert parsed[0].speaker == "SPEAKER_00"
