"""Tests for synth-COSER deterministic logic.

Network-heavy paths (HF dataset streaming, faster-whisper inference) are NOT
exercised here. These tests cover:
  - speaker grouping with min_clips,
  - turn sequence respects speaker alternation + duration target,
  - code-switch probability is honored,
  - audio assembly produces RTTM that sums to the speech-time,
  - music interlude replaces the right slice,
  - hypothesis evaluator grades correctly,
  - the FT-bench-rejection guard fires on synth_coser rows.
"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import soundfile as sf

from src.bench import assert_no_bench_rows
from src.bench.synth_coser import (
    SAMPLE_RATE,
    SynthCOSERConfig,
    Turn,
    assemble_recording,
    build_turn_sequence,
    group_by_speaker,
    inject_music_interlude,
    pick_speakers,
    write_rttm,
)
from src.bench.synth_coser_runner import (
    HypothesisResult,
    evaluate_hypotheses,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic clips on disk so assemble_recording can read them.
# ---------------------------------------------------------------------------


def _make_clip(
    tmp_path: Path, name: str, dur_s: float, transcript: str, speaker: str, language: str = "es"
) -> dict:
    n = int(round(dur_s * SAMPLE_RATE))
    audio = (np.random.default_rng(hash(name) & 0xFFFF).standard_normal(n) * 0.1).astype("float32")
    wav = tmp_path / f"{name}.wav"
    sf.write(str(wav), audio, SAMPLE_RATE)
    return {
        "segment_id": name,
        "audio_path": str(wav),
        "transcript": transcript,
        "duration_s": dur_s,
        "speaker_id": speaker,
        "accent": "castilian",
        "language": language,
    }


# ---------------------------------------------------------------------------
# group_by_speaker
# ---------------------------------------------------------------------------


def test_group_by_speaker_drops_below_min(tmp_path: Path):
    clips = [
        _make_clip(tmp_path, "a0", 2.0, "hola", "spkA"),
        _make_clip(tmp_path, "a1", 2.0, "hola", "spkA"),
        _make_clip(tmp_path, "b0", 2.0, "hola", "spkB"),  # only 1 clip → dropped
    ]
    pool = group_by_speaker(clips, min_clips=2)
    assert "spkA" in pool
    assert "spkB" not in pool


def test_pick_speakers_n_distinct(tmp_path: Path):
    clips = [_make_clip(tmp_path, f"x{i}", 1.0, "x", f"spk{i // 5}") for i in range(20)]
    pool = group_by_speaker(clips, min_clips=4)
    rng = random.Random(0)
    chosen = pick_speakers(pool, 3, rng)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3


def test_pick_speakers_raises_when_too_few(tmp_path: Path):
    clips = [_make_clip(tmp_path, f"x{i}", 1.0, "x", "only") for i in range(4)]
    pool = group_by_speaker(clips, min_clips=2)
    with pytest.raises(ValueError, match="Need 2 speakers"):
        pick_speakers(pool, 2, random.Random(0))


# ---------------------------------------------------------------------------
# build_turn_sequence
# ---------------------------------------------------------------------------


def _two_speaker_pool(tmp_path: Path) -> dict[str, list[dict]]:
    clips = []
    for i in range(8):
        clips.append(_make_clip(tmp_path, f"a{i}", 2.5, f"frase a {i}", "spkA"))
        clips.append(_make_clip(tmp_path, f"b{i}", 2.5, f"frase b {i}", "spkB"))
    return group_by_speaker(clips, min_clips=4)


def test_turn_sequence_alternates_speakers(tmp_path: Path):
    pool = _two_speaker_pool(tmp_path)
    rng = random.Random(0)
    speakers = pick_speakers(pool, 2, rng)
    turns = build_turn_sequence(
        speakers,
        pool,
        secondary_pools={},
        target_duration_s=20.0,
        silence_range=(0.5, 0.5),
        code_switch_prob=0.0,
        rng=rng,
    )
    # Speakers must strictly alternate.
    for prev, curr in zip(turns, turns[1:]):
        assert prev.speaker_id != curr.speaker_id


def test_turn_sequence_reaches_target_duration(tmp_path: Path):
    pool = _two_speaker_pool(tmp_path)
    rng = random.Random(0)
    speakers = pick_speakers(pool, 2, rng)
    target = 30.0
    turns = build_turn_sequence(
        speakers,
        pool,
        secondary_pools={},
        target_duration_s=target,
        silence_range=(0.5, 0.5),
        code_switch_prob=0.0,
        rng=rng,
    )
    total = sum(t.duration_s for t in turns) + 0.5 * (len(turns) - 1)
    assert total >= target


def test_turn_sequence_honors_code_switch_probability(tmp_path: Path):
    primary_pool = _two_speaker_pool(tmp_path)
    secondary = [
        _make_clip(tmp_path, f"ca_{i}", 2.0, "bon dia", "ca_spk", language="ca") for i in range(20)
    ]
    rng = random.Random(0)
    speakers = pick_speakers(primary_pool, 2, rng)
    turns = build_turn_sequence(
        speakers,
        primary_pool,
        secondary_pools={"ca": secondary},
        target_duration_s=200.0,
        silence_range=(0.0, 0.0),
        code_switch_prob=1.0,  # all turns should be CA
        rng=rng,
    )
    assert all(t.language == "ca" for t in turns)


def test_turn_sequence_no_code_switch_when_prob_zero(tmp_path: Path):
    primary_pool = _two_speaker_pool(tmp_path)
    secondary = [
        _make_clip(tmp_path, f"ca_{i}", 2.0, "bon dia", "ca_spk", language="ca") for i in range(8)
    ]
    rng = random.Random(0)
    speakers = pick_speakers(primary_pool, 2, rng)
    turns = build_turn_sequence(
        speakers,
        primary_pool,
        secondary_pools={"ca": secondary},
        target_duration_s=20.0,
        silence_range=(0.0, 0.0),
        code_switch_prob=0.0,
        rng=rng,
    )
    assert all(t.language == "es" for t in turns)


# ---------------------------------------------------------------------------
# assemble_recording
# ---------------------------------------------------------------------------


def test_assemble_recording_produces_consistent_durations(tmp_path: Path):
    pool = _two_speaker_pool(tmp_path)
    rng = random.Random(0)
    speakers = pick_speakers(pool, 2, rng)
    turns = build_turn_sequence(
        speakers,
        pool,
        secondary_pools={},
        target_duration_s=15.0,
        silence_range=(0.5, 0.5),
        code_switch_prob=0.0,
        rng=rng,
    )
    audio, segments, lines, timeline = assemble_recording(turns, silence_range=(0.5, 0.5), rng=rng)
    audio_dur = audio.shape[0] / SAMPLE_RATE
    speech_dur = sum(s["end_s"] - s["start_s"] for s in segments)
    silence_dur = sum(t["t_end"] - t["t_start"] for t in timeline if t["kind"] == "silence")
    # Total = speech + silence (within rounding).
    assert abs(audio_dur - (speech_dur + silence_dur)) < 0.05
    assert len(lines) == len(turns)


def test_rttm_segments_do_not_overlap(tmp_path: Path):
    pool = _two_speaker_pool(tmp_path)
    rng = random.Random(0)
    speakers = pick_speakers(pool, 2, rng)
    turns = build_turn_sequence(
        speakers,
        pool,
        secondary_pools={},
        target_duration_s=15.0,
        silence_range=(0.3, 0.3),
        code_switch_prob=0.0,
        rng=rng,
    )
    _, segments, _, _ = assemble_recording(turns, silence_range=(0.3, 0.3), rng=rng)
    for prev, curr in zip(segments, segments[1:]):
        assert curr["start_s"] >= prev["end_s"]


def test_assemble_recording_first_turn_no_leading_silence(tmp_path: Path):
    pool = _two_speaker_pool(tmp_path)
    rng = random.Random(0)
    speakers = pick_speakers(pool, 2, rng)
    turns = build_turn_sequence(
        speakers,
        pool,
        secondary_pools={},
        target_duration_s=10.0,
        silence_range=(0.5, 0.5),
        code_switch_prob=0.0,
        rng=rng,
    )
    _, segments, _, _ = assemble_recording(turns, silence_range=(0.5, 0.5), rng=rng)
    assert segments[0]["start_s"] == 0.0


# ---------------------------------------------------------------------------
# inject_music_interlude
# ---------------------------------------------------------------------------


def test_inject_music_interlude_replaces_correct_slice():
    audio = np.ones(SAMPLE_RATE * 10, dtype="float32") * 0.5
    out, interval = inject_music_interlude(audio, position_s=3.0, duration_s=2.0)
    assert interval[0] == pytest.approx(3.0, abs=1e-6)
    assert interval[1] == pytest.approx(5.0, abs=1e-6)
    # Outside the interlude the signal is unchanged.
    assert np.allclose(out[: SAMPLE_RATE * 3], 0.5)
    assert np.allclose(out[SAMPLE_RATE * 5 :], 0.5)
    # Inside, it should NOT be the original value (music replaces it).
    assert not np.allclose(out[SAMPLE_RATE * 3 : SAMPLE_RATE * 5], 0.5)


def test_inject_music_interlude_clamps_to_audio_length():
    audio = np.zeros(SAMPLE_RATE * 5, dtype="float32")
    out, interval = inject_music_interlude(audio, position_s=4.0, duration_s=10.0)
    assert interval[1] <= 5.0


# ---------------------------------------------------------------------------
# write_rttm
# ---------------------------------------------------------------------------


def test_write_rttm_round_trip(tmp_path: Path):
    segs = [
        {"start_s": 0.5, "end_s": 1.5, "speaker": "spkA"},
        {"start_s": 2.0, "end_s": 3.5, "speaker": "spkB"},
    ]
    path = tmp_path / "rec.rttm"
    write_rttm("rec", segs, path)
    text = path.read_text(encoding="utf-8")
    assert "SPEAKER rec 1 0.500 1.000 <NA> <NA> spkA <NA> <NA>" in text
    assert "SPEAKER rec 1 2.000 1.500 <NA> <NA> spkB <NA> <NA>" in text


# ---------------------------------------------------------------------------
# evaluate_hypotheses (runner)
# ---------------------------------------------------------------------------


def _record(**kwargs) -> dict:
    base = {
        "record_id": "r0",
        "primary_language": "es",
        "code_switch_count": 0,
        "has_music_interlude": False,
        "n_speakers": 2,
        "duration_s": 60.0,
        "raw_wer": 0.10,
        "normalized_wer": 0.08,
        "hyp_text": "hola mundo",
        "accent_majority": "castilian",
        "der": None,
        "der_snapped": None,
    }
    base.update(kwargs)
    return base


def test_h1_passes_with_low_wer():
    records = [_record(record_id=f"r{i}", normalized_wer=0.10) for i in range(3)]
    _, hyps = evaluate_hypotheses(records)
    h1 = next(h for h in hyps if h.name == "H1")
    assert h1.passed


def test_h1_fails_with_high_wer():
    records = [_record(record_id=f"r{i}", normalized_wer=0.50) for i in range(3)]
    _, hyps = evaluate_hypotheses(records)
    h1 = next(h for h in hyps if h.name == "H1")
    assert not h1.passed


def test_h2_detects_code_switch_degradation():
    records = [
        _record(record_id="es_0", normalized_wer=0.10, code_switch_count=0),
        _record(record_id="es_1", normalized_wer=0.10, code_switch_count=0),
        _record(record_id="cs_0", normalized_wer=0.30, code_switch_count=3),
        _record(record_id="cs_1", normalized_wer=0.25, code_switch_count=2),
    ]
    _, hyps = evaluate_hypotheses(records)
    h2 = next(h for h in hyps if h.name == "H2")
    assert h2.passed  # 0.275 - 0.10 = 0.175 ≥ 0.05


def test_h2_skipped_when_no_code_switch_data():
    records = [_record(record_id=f"r{i}", code_switch_count=0) for i in range(3)]
    _, hyps = evaluate_hypotheses(records)
    assert not any(h.name == "H2" for h in hyps)


def test_h3_detects_accent_drift():
    records = [
        _record(record_id="cas_0", accent_majority="castilian", normalized_wer=0.05),
        _record(record_id="cas_1", accent_majority="castilian", normalized_wer=0.07),
        _record(record_id="and_0", accent_majority="andalusian", normalized_wer=0.20),
        _record(record_id="and_1", accent_majority="andalusian", normalized_wer=0.18),
    ]
    _, hyps = evaluate_hypotheses(records)
    h3 = next(h for h in hyps if h.name == "H3")
    assert h3.passed


def test_h_sd_passes_with_low_der():
    records = [
        _record(record_id="r0", der=0.20, der_snapped=0.15),
        _record(record_id="r1", der=0.25, der_snapped=0.18),
    ]
    aggregates, hyps = evaluate_hypotheses(records)
    h_sd = next(h for h in hyps if h.name == "H_SD")
    assert h_sd.passed
    h_snap = next(h for h in hyps if h.name == "H_SNAP")
    assert h_snap.passed
    # snap delta = 0.06 average → ≥ 0.02 threshold ✓
    assert aggregates["snap_delta_der"] == pytest.approx(0.06)


def test_hypothesis_result_fields():
    h = HypothesisResult(
        name="H1", description="x", threshold="≤0.25", measured="0.10", passed=True
    )
    assert h.passed
    assert h.threshold == "≤0.25"


# ---------------------------------------------------------------------------
# FT-rejection guard
# ---------------------------------------------------------------------------


def test_synth_coser_source_blocked_from_ft():
    import pandas as pd

    df = pd.DataFrame([{"audio_path": "x.wav", "source": "synth_coser"}])
    with pytest.raises(ValueError, match="(?i)external audio is forbidden"):
        assert_no_bench_rows(df)


def test_config_defaults_are_safe():
    cfg = SynthCOSERConfig()
    assert cfg.target_duration_min > 0
    assert cfg.n_speakers >= 1
    assert 0 <= cfg.code_switch_probability <= 1
    # cv_dataset is now derived from SOURCE_REGISTRY (open-license sources only).
    assert "/" in cfg.cv_dataset


def test_source_registry_covers_es_ca_gl():
    from src.bench.synth_coser import SOURCE_REGISTRY

    for lang in ("es", "ca", "gl"):
        assert lang in SOURCE_REGISTRY
        entry = SOURCE_REGISTRY[lang]
        assert "/" in entry["dataset"]
        assert "transcript_key" in entry


def test_turn_dataclass_holds_what_we_expect():
    t = Turn(speaker_id="A", language="es", audio_path="x.wav", transcript="hola", duration_s=2.0)
    assert t.duration_s == 2.0
    assert t.transcript == "hola"
